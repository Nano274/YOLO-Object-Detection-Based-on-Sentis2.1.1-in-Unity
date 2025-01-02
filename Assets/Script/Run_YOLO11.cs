using System.Collections;
using System.Collections.Generic;
using System.Threading.Tasks;
using Unity.Sentis;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Video;
using Lays = Unity.Sentis.Layers;
using System.IO;
using FF = Unity.Sentis.Functional;

public class Run_YOLO11 : MonoBehaviour
{
    // モデルに関するセッティング
    public ModelAsset asset;
    public TextAsset labelsAsset;
    private string[] labels;
    private const int numClasses = 80;
    // 静的なテンソル（YOLO後処理用）
    Tensor<float> centersToCorners;

    // ！モデル実行設定　
    private Worker engine;

    // 表示関連設定
    // RawImage（シーンに配置）
    public RawImage displayImage;
    private Transform displayLocation;
    // バウンディングボックススプライト、ボックスのテクスチャ
    public Sprite borderSprite;
    public Texture2D borderTexture;
    // ラベルフォント
    public Font font;

    // 使用するバックエンド（GPU/CPU）
    const BackendType backend = BackendType.GPUCompute;

    // 入力に関する
    const string videoName = "video.mp4";
    private VideoPlayer video;
    private RenderTexture targetRT;

    // モデル入力画像幅 モデル入力画像高さ(YOLO専用入力：640x640)
    private const int imageWidth = 640;
    private const int imageHeight = 640;

    // NMSとIoUに関するセッティング、検出の上限
    [SerializeField, Range(0, 1)] float iouThreshold = 0.5f;
    [SerializeField, Range(0, 1)] float scoreThreshold = 0.5f;

    // バウンディングボックスプール
    List<GameObject> boxPool = new();
    // ラベルカラー辞書、ランダムカラー生成用
    private Dictionary<string, Color> labelColorMap = new Dictionary<string, Color>();
    private System.Random random = new System.Random();

    // バウンディングボックスデータ構造体
    public struct BoundingBox
    {
        public float centerX;
        public float centerY;
        public float width;
        public float height;
        public string label;
        public float score;
    }

    void Start()
    {
        //フレームの設定
        Application.targetFrameRate = 60;
        Screen.orientation = ScreenOrientation.LandscapeLeft;

        //モデルロード
        LoadModel();

        //ラベル読み込み
        labels = labelsAsset.text.Split('\n');
        //表示位置設定
        targetRT = new RenderTexture(imageWidth, imageHeight, 0);
        displayLocation = displayImage.transform;

        //毎フレームの動画を画像として入力
        SetupInput();

        //画像のレンダリング用かな
        if (borderSprite == null)
        {
            borderSprite = Sprite.Create(borderTexture, new Rect(0, 0, borderTexture.width, borderTexture.height), new Vector2(borderTexture.width / 2, borderTexture.height / 2));
        }
    }

    void LoadModel()
    {

        // モデルをロード
        Model model1 = ModelLoader.Load(asset);

        // YOLO後処理用テンソル
        centersToCorners = new Tensor<float>(new TensorShape(4, 4),
        new float[]
        {
                    1,      0,      1,      0,
                    0,      1,      0,      1,
                    -0.5f,  0,      0.5f,   0,
                    0,      -0.5f,  0,      0.5f
        });

        //計算グラフ作成
        FunctionalGraph graph = new();

        //テンソルを入力、出力から「バンティングブックス座標」「分類スコア」「最高スコアとID」
        //「座標変更」「NMS」「バンティングブックスとIDを選択」
        FunctionalTensor input = graph.AddInputs(model1)[0];
        FunctionalTensor output = Functional.Forward(model1, input)[0];
        FunctionalTensor boxCoords = output[0, 0..4, ..].Transpose(0, 1);        //shape=(8400,4)
        FunctionalTensor allScores = output[0, 4.., ..];                         //shape=(80,8400)
        FunctionalTensor scores = Functional.ReduceMax(allScores, 0);        //shape=(8400)
        FunctionalTensor classIDs = Functional.ArgMax(allScores, 0);                          //shape=(8400) 
        FunctionalTensor boxCorners = Functional.MatMul(boxCoords, Functional.Constant(centersToCorners));
        FunctionalTensor indices = Functional.NMS(boxCorners, scores, iouThreshold, scoreThreshold);           //shape=(N)
        FunctionalTensor indices2 = indices.Unsqueeze(-1).BroadcastTo(new int[] { 4 });//shape=(N,4)
        FunctionalTensor coords = Functional.Gather(boxCoords, 0, indices2);                  //shape=(N,4)
        FunctionalTensor labelIDs = Functional.Gather(classIDs, 0, indices);                  //shape=(N)
        FunctionalTensor selectedScores = Functional.Gather(scores, 0, indices);                  // shape=(N)
        //計算グラフを新たなモデルとして作り
        Model runtimeModel = graph.Compile(coords, labelIDs, selectedScores);

        // 実行エンジン作成
        engine = new(runtimeModel, BackendType.GPUCompute);
    }

    void SetupInput()
    {
        video = gameObject.AddComponent<VideoPlayer>();
        video.renderMode = VideoRenderMode.APIOnly;
        video.source = VideoSource.Url;
        video.url = Path.Join(Application.streamingAssetsPath, videoName);
        video.isLooping = true;
        video.Play();
    }

    private void Update()
    {
        // 予測を実行
        Predict();

        // 用がない仕組み
        if (Input.GetKeyDown(KeyCode.Escape))
        {
            Application.Quit();
        }
    }

    public void Predict()
    {
        ClearAnnotations();

        // 動画のフレームを取得
        if (video && video.texture)
        {
            float aspect = video.width * 1f / video.height;
            Graphics.Blit(video.texture, targetRT, new Vector2(1f / aspect, 1), new Vector2(0, 0));
            displayImage.texture = targetRT;
        }
        else return;

        // 画像をテンソルに変換
        Tensor<float> input = TextureConverter.ToTensor(targetRT, imageWidth, imageHeight, 3);
        // テンソルをモデルに入力
        engine.Schedule(input);
        // モデルを実行(通过 PeekOutput 方法获取模型的推理输出张量)
        // 参数 "output_0" 和 "output_1" 是模型输出的名称（或索引）
        var output = engine.PeekOutput("output_0") as Tensor<float>;
        var labelIDs = engine.PeekOutput("output_1") as Tensor<int>;
        var selectedScores = engine.PeekOutput("output_2") as Tensor<float>;

        // CPUにデータをコピー(何となく、このバッジョンのSentisはGPUからテンソルを読み込めまない。。)
        // 一番面倒な部分だと思う
        Tensor<float> cpuOutput = output.ReadbackAndClone();
        Tensor<int> cpulabelIDs = labelIDs.ReadbackAndClone();
        Tensor<float> cpuScores = selectedScores.ReadbackAndClone();

        // テンソルを解放
        input?.Dispose();
        output?.Dispose();
        labelIDs.Dispose();
        selectedScores.Dispose();

        // 画像のサイズを取得
        float displayWidth = displayImage.rectTransform.rect.width;
        float displayHeight = displayImage.rectTransform.rect.height;
        // 画像のスケールを計算
        float scaleX = displayWidth / imageWidth;
        float scaleY = displayHeight / imageHeight;
        // 検出されたバンティングブックスの数を取得
        int boxesFound = cpuOutput.shape[0];

        // バウンディングボックス描画
        for (int n = 0; n < Mathf.Min(boxesFound, 200); n++)
        {
            var box = new BoundingBox
            {
                centerX = cpuOutput[n, 0] * scaleX - displayWidth / 2,
                centerY = cpuOutput[n, 1] * scaleY - displayHeight / 2,
                width = cpuOutput[n, 2] * scaleX,
                height = cpuOutput[n, 3] * scaleY,
                label = labels[cpulabelIDs[n]],
                score = (float)cpuScores[n],
            };
            DrawBox(box, n, displayHeight * 0.05f);
        }
        //解放
        cpuOutput?.Dispose();
        cpulabelIDs?.Dispose();
        cpuScores?.Dispose();
    }

    public void DrawBox(BoundingBox box, int id, float fontSize)
    {
        // プールから取得
        GameObject panel;
        if (id < boxPool.Count)
        {
            panel = boxPool[id];
            panel.SetActive(true);
        }
        else
        {
            panel = CreateNewBox(GetColorForLabel(box.label)); // 获取随机颜色
        }

        Image img = panel.GetComponent<Image>(); // 获取 Image 组件
        Color color1 = GetColorForLabel(box.label);
        color1.a = 0.5f; // 设置透明度
        img.color = color1; // 设置颜色
        // Set box position
        panel.transform.localPosition = new Vector3(box.centerX, -box.centerY);
        // Set box size
        RectTransform rt = panel.GetComponent<RectTransform>();
        rt.sizeDelta = new Vector2(box.width, box.height);
        // Set label text
        var label = panel.GetComponentInChildren<Text>();
        label.color = GetColorForLabel(box.label);
        label.horizontalOverflow = HorizontalWrapMode.Overflow; // 水平方向允许溢出
        label.verticalOverflow = VerticalWrapMode.Overflow;     // 垂直方向允许溢出
        label.text = $"{box.label} {box.score:F2}"; // 格式化字符串，显示标签和置信度
        label.fontSize = (int)fontSize;
    }

    public GameObject CreateNewBox(Color color)
    {
        // Create the box and set image
        var panel = new GameObject("ObjectBox");
        panel.AddComponent<CanvasRenderer>();
        Image img = panel.AddComponent<Image>();
        img.color = color; // 使用传入的颜色
        img.sprite = borderSprite;
        img.type = Image.Type.Sliced;
        // 边界框是通过 UI （SetParent）的方式叠加在 displayImage 上显示的
        panel.transform.SetParent(displayLocation, false);

        // Create the label
        var text = new GameObject("ObjectLabel");
        text.AddComponent<CanvasRenderer>();
        text.transform.SetParent(panel.transform, false);
        Text txt = text.AddComponent<Text>();
        txt.font = font;
        txt.color = color; // 设置与边界框一致的颜色
        txt.fontSize = 40;
        txt.horizontalOverflow = HorizontalWrapMode.Overflow;

        RectTransform rt2 = text.GetComponent<RectTransform>();
        rt2.offsetMin = new Vector2(20, rt2.offsetMin.y);
        rt2.offsetMax = new Vector2(0, rt2.offsetMax.y);
        rt2.offsetMin = new Vector2(rt2.offsetMin.x, 0);
        rt2.offsetMax = new Vector2(rt2.offsetMax.x, 30);
        rt2.anchorMin = new Vector2(0, 0);
        rt2.anchorMax = new Vector2(1, 1);

        boxPool.Add(panel);
        return panel;
    }

    private Color GetColorForLabel(string label)
    {
        // 如果字典中已经有这个标签的颜色，直接返回
        if (labelColorMap.ContainsKey(label))
            return labelColorMap[label];

        // 如果没有，为该标签生成一个随机颜色
        Color randomColor = new Color((float)random.NextDouble(), (float)random.NextDouble(), (float)random.NextDouble());
        labelColorMap[label] = randomColor; // 保存到字典中
        return randomColor;
    }

    private Color GetLabelColor(int classIndex)
    {
        return new Color(UnityEngine.Random.value, UnityEngine.Random.value, UnityEngine.Random.value); // 使用随机颜色示例
    }

    public void ClearAnnotations()
    {
        foreach (var box in boxPool)
        {
            box.SetActive(false);
        }
    }
}
from flask import Flask, render_template
from flask import request, jsonify
import xgboost as xgb
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='xgboost')

# 初始化 Flask 应用
app = Flask(__name__)

# 加载已训练好的XGBoost模型
loaded_model = xgb.XGBClassifier()
loaded_model.load_model("xgb_model.json")

# 首页路由，指向 index.html
@app.route('/')
def prediction_system():
    return render_template('index.html')  # 渲染 templates 文件夹下的 index.html

@app.route("/predict", methods=["POST"])
def predict():
    # 获取请求数据
    data = request.get_json()

    # 将数据转换为合适的格式供模型预测
    features_input = np.array([[data["HighBP"], data["HighChol"], data["BMI"], data["Smoker"],
                          data["Stroke"], data["HeartDiseaseorAttack"], data["PhysActivity"], data["HvyAlcoholConsump"],
                          data["NoDocbcCost"], data["GenHlth"], data["MentHlth"],
                          data["PhysHlth"], data["DiffWalk"], data["Age"],data["Education"],data["Income"]]])
    
    features_input = np.array([[float(data[key]) for key in data]])


    # 将单条记录转换为二维数组，reshape(1, -1) 使其形状符合模型输入
    features_input = features_input.reshape(1, -1)

    # 预测结果
    prediction = loaded_model.predict(features_input)
    probability = loaded_model.predict_proba(features_input)[0][prediction[0]] * 100

    # 输出预测结果
    print(f"Predicted class for the input record: {prediction[0]}")
    
    # 返回预测结果
    print(features_input)

    # 构建 JSON 响应
    result = {
        "prediction": str(prediction[0]),  # 输出预测结果（类别）
        "probability": str(probability)  # 输出预测为该类别的概率，转换为百分比
    }

    print(result)

    return jsonify(result)


# 文档路由
@app.route('/apidoc')
def apidoc():
    return render_template('API Document.html')  # 渲染 templates 文件夹下的 API Document.html
    

# 启动应用
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)

import yaml
#from src.data.data_processing import load_and_preprocess
#from src.data.feature_eng import extract_features
#from src.utils.evaluation import evaluate

# 모델 모드별 import
#from src.models import supervised, unsupervised, transfer

def main():
    
    # 1. 설정 불러오기
    with open('config.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    print(cfg)
    # 데이터 로드
    #raw_data = load_and_preprocess(r"C:\Dev\KorailTM\test.csv")
    #print(raw_data.head())
    #print("test")

    # 데이터 전처리
#    processed_data = preprocess_data(raw_data)
#    features = extract_features(processed_data)
    
    # 모델 학습
#    model = train_model(features, target="failure_label")
    
    # 예측
#    predictions = predict(model, features)
    
    # 유지보수 최적화
#    maintenance_schedule = optimize_maintenance(predictions)
#    print("최적 유지보수 일정:", maintenance_schedule)

if __name__ == "__main__":
    main()
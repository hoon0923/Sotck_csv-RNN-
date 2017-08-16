주류업체 주식데이터(open , high , low , volume , close) 와 날씨데이터(평균 기온)를 사용하여 RNN학습시킨 후 주식데이터값(Ex : close)예측

전체 데이터의 70%는 트레이닝 시키고 30%를 테스트함.

EX) input -> open , high , low , volume , 평균 기온 
    output -> close
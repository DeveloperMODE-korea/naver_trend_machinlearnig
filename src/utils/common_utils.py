import numpy as np

class CommonUtils:
    """공통 유틸리티 클래스"""
    
    @staticmethod
    def get_user_input(prompt, default=None, input_type=str):
        """사용자 입력을 받는 헬퍼 함수"""
        try:
            if default is not None:
                user_input = input(f"{prompt} (기본값: {default}): ").strip()
                if not user_input:
                    return default
            else:
                user_input = input(f"{prompt}: ").strip()
            
            if input_type == int:
                return int(user_input)
            elif input_type == float:
                return float(user_input)
            elif input_type == bool:
                return user_input.lower() in ['y', 'yes', 'true', '1']
            else:
                return user_input
        except ValueError:
            if default is not None:
                print(f"유효하지 않은 입력입니다. 기본값 {default}를 사용합니다.")
                return default
            else:
                raise ValueError("유효하지 않은 입력입니다.")

    @staticmethod
    def create_sequences(data, seq_length):
        """LSTM을 위한 시퀀스 데이터를 생성합니다"""
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data[i:i+seq_length]
            y = data[i+seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    @staticmethod
    def ensemble_weighted(pred_list, weights):
        """가중평균 앙상블을 수행합니다"""
        arrs = [np.array(a) for a in pred_list if a is not None]
        min_len = min([len(a) for a in arrs])
        arrs = [a[:min_len] for a in arrs]
        weights = np.array(weights)[:len(arrs)]
        weights = weights / weights.sum()
        arrs = np.stack(arrs, axis=0)
        return np.average(arrs, axis=0, weights=weights) 
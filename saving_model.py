from model import MODEL
import warnings
warnings.filterwarnings('ignore')

model_name = 'retinanet'

model = MODEL(model_name=model_name)
model.save_model(f'../DL_model/{model_name}.pth')
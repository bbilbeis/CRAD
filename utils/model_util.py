from api.model.resnet import  customized_resnet18

def create_model(model_name,class_num):
    model = None
    if model_name == "resnet18":
        model = customized_resnet18(class_num=class_num)
    return model
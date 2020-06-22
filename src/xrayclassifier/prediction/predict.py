from PIL import Image
from torch.autograd import Variable

class Prediction:
    def __init_(self, image_path, model, transforms, classes):
        self.image_path = image_path
        self.model = model
        self.transforms = transforms
        self.classes = classes

    def predict_class(self):
        # print("Starting prediction")
        image = Image.open(self.image_path).convert('RGB')

        # define transforms
        transformation = self.transforms['val']

        # transform the image
        image_tensor = transformation(image).float()

        # Add an extra batch dimension since pytorch treats all images as batches
        image_tensor = image_tensor.unsqueeze_(0)

        if torch.cuda.is_available():
            image_tensor.cuda()

        # Turn the input into a Variable
        input = Variable(image_tensor)

        # Predict the class of the image
        output = self.model(input)

        index = output.data.numpy().argmax()

        return self.classes[index]
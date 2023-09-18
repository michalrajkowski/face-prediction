# Gender Prediction with TensorFlow

This repository contains a gender prediction model built using TensorFlow. The model predicts the gender (female or male) based on input face images. The model has an accuracy rate of approximately 91%. 

## Getting Started

1. Clone this repository to your local machine:

```bash
git clone https://github.com/michalrajkowski/face-prediction.git
cd face-prediction
```

To mention the requirements and their versions in your README.md using Markdown, you can use a code block or inline code formatting. Here's how you can do it:

2. Make sure you have all required libraries installed

You can check if you have all the required libraries installed by running the following command:

```bash
pip install -r requirements.txt
```
3. Organize your face images in the `input images` directory. 

The neural network performs at its peak when provided with forward-facing images of faces without any tilting.

![Face images](https://media.discordapp.net/attachments/1153463839053062247/1153466912957812736/image.png)

Above images originate from [Pexels](https://www.pexels.com/search/face/).

4. Run the prediction script:

```bash
python3 main.py
```

The predictions will be saved as images in the `output images` directory, and a `predictions.txt` file will contain the results in the text form.

## Results

### images 

Below is an example of how the predictions are saved as original images with their names, predicted gender, and certainty percentage:

![Example prediction](https://media.discordapp.net/attachments/1153463839053062247/1153467370111783023/image.png)

### text file
The predictions in the `predictions.txt` file appear to be saved in the following format:

```
[Image Name] [Predicted Gender] [Certainty Percentage]
```
In this format:

- `[Image Name]`: Represents the name of the image for which the gender was predicted.
- `[Predicted Gender]`: Represents the predicted gender (e.g., "Male" or "Female").
- `[Certainty Percentage]`: Represents the certainty level of the prediction as a percentage (e.g., 100.0% indicates high confidence).

## License

This project is licensed under the MIT License 
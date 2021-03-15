## 3D_DEN: Open-ended 3D Object Recognition using Dynamically Expandable Networks
![alt text](model_arch.png)
  An overview of the proposed3D_DENmodel: Initially, three representative views are chosen from a set of multi-view images for a given 3D object.Then, each of them is converted to a single channel (grey-scale) image and later merged to form a 3-channel image. Now, this image is fed to a pre-trainednetwork, and the extracted features are flattened. Finally, we attach two DEN layers to the model which give the output.
  
  

- This project hosts the code for our [**3D_DEN** paper](https://arxiv.org/pdf/2009.07213.pdf) and [report](https://fse.studenttheses.ub.rug.nl/23621/1/SJ_Graduation_Thesis_Final_submission.pdf)
- Major parent papers that inspired our work are : [DEN](https://openreview.net/pdf?id=Sk7KsfW0-) and [OrthographicNet](https://arxiv.org/pdf/1902.03057.pdf)
- Video demo using a real-time robot can be found [here](https://youtu.be/tf4trRMyQ0Y).

## Requirements:
- Python 3.6
- Kindly create a virtual environment using requirements.txt file to run the code  
- **Note:** For Offline Evaluation using GridSearch, use Tensorflow and Tensorboard version: 2.3.0.

## Authors: 
[Sudhakaran Jain](https://sudhakaranjain.github.io/) and [Hamidreza Kasaei](https://hkasaei.github.io/)  
Work done while at [RUG](https://www.rug.nl/).

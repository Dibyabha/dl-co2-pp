# dl-co2-pp
This is the official repository for the paper "Improving Power Plant CO2 Emission Estimation with Deep Learning and Satellite/Simulated Data" and has been submitted to the NeurIPS 2024 conference (NeurIPS 2024 Workshop on Tackling Climate Change with Machine Learning)

The link to the paper: https://www.climatechange.ai/papers/neurips2024/25

# Abstract 
CO2 emissions from power plants, as significant super emitters, contribute substantially to global warming. Accurate quantification of these emissions is crucial for effective climate mitigation strategies. While satellite-based plume inversion offers a promising approach, challenges arise from data limitations and the complexity of atmospheric conditions. This study addresses these challenges by (a) expanding the available dataset through the integration of NO2 data from Sentinel-5P, generating continuous XCO2 maps, and incorporating real satellite observations from OCO-2/3 for over 71 power plants in data-scarce regions; and (b) employing a customized U-Net model capable of handling diverse spatio-temporal resolutions for emission rate estimation. Our results demonstrate significant improvements in emission rate accuracy compared to previous methods (Ref). By leveraging this enhanced approach, we can enable near real-time, precise quantification of major CO2 emission sources, supporting environmental protection initiatives and informing regulatory frameworks. 

# Project Overview
This project introduces an encoder-decoder architecture to enhance the accuracy of the emission rates estimation. Along with that, we have also integrated satellite data with simulated data to create a new dataset to evaluate our model and enhance the generalizability of the model. 

Our codes are in Python, particularly in Tensorflow

The figure shows the proposed methodology that we followed:

![thumb](https://github.com/user-attachments/assets/4810d1ce-c7f0-450a-b9c1-14c5bfe186bc)


### To employ the codes, follow the steps below
Steps to follow to replicate this work:
1. Simulated -> run org_model_train_eval.py
2. Simulated -> run shuf_model_train_eval.py
3. Simulated -> run sim_eval.py
4. Simulated -> run error_calc.py
5. Satellite -> run sat_cur.py
6. Satellite -> run sat_eval.py
7. Satellite -> run error_calc.py
8. Combined -> run comb.py
9. Combined -> run model.py
10. Combined -> run error_calc.py
11. Additionally, EDA -> eda.py 

# Citation
Please cite our paper if you use our work
<pre>
@misc{deb2025improvingpowerplantco2,
      title={Improving Power Plant CO2 Emission Estimation with Deep Learning and Satellite/Simulated Data}, 
      author={Dibyabha Deb and Kamal Das},
      year={2025},
      eprint={2502.02083},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.02083}, 
}
</pre>
# Support
Feel free to contact: dibyabhadeb@gmail.com

# References

1. Jeremy Stott, Hairong Bau, and Terence L. van Zyl (2023). *Training a U-Net Model for Brain Tumour Segmentation Using Federated Learning*. Original project and associated report: https://github.com/jeremyscodes/Training-a-U-Net-Model-for-Brain-Tumour-Segmentation-Using-Federated-Learning . This motivated the reproduction and comparison study; original repository inspected at commit 9255825669277f0c64d80f970cf40b8de83d3901.
2. Olaf Ronneberger, Philipp Fischer, and Thomas Brox (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. https://arxiv.org/abs/1505.04597 . Source of the encoder-decoder architecture with skip connections.
3. H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data*. AISTATS, PMLR 54:1273–1282. https://proceedings.mlr.press/v54/mcmahan17a.html . Source of federated averaging.
4. Amber L. Simpson et al. (2019). *A large annotated medical image dataset for the development and evaluation of segmentation algorithms*. https://arxiv.org/abs/1902.09063 . Medical Segmentation Decathlon data source; Task01_BrainTumour derives from BraTS. Consult the dataset documentation for its underlying BraTS citations too.
5. Medical Segmentation Decathlon public dataset registry: https://registry.opendata.aws/msd/ . Dataset license: CC BY-SA 4.0. Retain required attribution and follow the dataset's terms when redistributing data or adaptations. No medical images are included in this source package.

Software dependencies, including TensorFlow, tf-keras, NumPy, NiBabel, Matplotlib, and pytest, retain their own licenses. Record the actual versions from each run's configuration.

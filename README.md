
![nori](https://github.com/rkarmaka/NoRI/assets/16607915/ab24c59f-3ba9-4171-b6d6-e3ae4a527a71)

# NoRI
Normalized Raman Imaging (NoRI) is a high-resolution imaging technique that measures protein, lipid, and water concentrations in cells and tissues by normalizing Raman signals. This method provides detailed insights into tissue states and the effects of various conditions by analyzing the protein and lipid content. In this project, we use NoRI to examine kidney images, focusing on the tubules as the main structures, along with substructures such as the nucleus, brush border, and lumen. Additionally, different tubule markers like Lectin, Uro, and AQP2 are analyzed to classify tubule types. While NoRI excels at imaging intricate structures, analyzing these complex images remains a challenge due to their delicate architecture.

The objective of this project is to develop advanced image analysis methods to address these challenges. The tasks are divided into three main components: segmenting the tubules, segmenting the substructures, and measuring the amount of lipid and protein per tubule. By achieving these goals, we aim to enhance the understanding of kidney tissue states and improve the ability to analyze the effects of various conditions on these tissues. This project will leverage NoRI's capabilities to provide a comprehensive and quantitative analysis of kidney images, ultimately contributing to better diagnostic and research outcomes.


## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Collaboration](#sollaboration)
- [Contact](#contact)
- [References](#references)

## Installation

```bash
git clone https://github.com/rkarmaka/NoRI.git
cd NoRI
```

## Usage
First, clone the repo to get started.

```bash
git clone https://github.com/rkarmaka/NoRI.git
cd NoRI
```

* Use ```metadata_reader.py``` to read the metadata and image intensity information for all images
* Use ```segment_tubule.py``` to segment tubules
* Use ```nuclei_bb.ilp``` file to segment the nucleus, brush border, and lumen using [Ilastik](https://www.ilastik.org)
* Use ```glue.py``` to create a segmentation mask for the cytoplasm only and then measure the intensity. It also classifies tubule types using the IF channels.

## License

## Collaboration
This project was a collaboration between the [Kirschner lab](http://kirschnerlab.org) and [Image Analysis Collaboratory](http://iac.hms.harvard.edu).

## Contact
TBD.

## References / Relavant Publications
TBA.

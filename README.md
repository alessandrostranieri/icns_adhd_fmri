# ADHD Data Experiments

## Set-up

In order to run the experiments, it is assumed that you've downloaded time series files and create a simple directory structure. Here the steps to do that:

1. Create a directory dedicated to the experiment data
2. Configure the directory by editing the `data_root` property in the `config.ini` file
3. The time series for the AAL and CC200 atlases can be downloaded from this [link](https://www.nitrc.org/plugins/mwiki/index.php/neurobureau:AthenaPipeline#Extracted_Time_Courses). To download the files, one needs to register on the website.
4. Under `data_root` create `Train\TC` directory 
5. Under `data_root\Train\TC` extract time series archives

The final directory structure should look similar to this

```
.
├── Test
│   └── TC
│       ├── Brown
│       ├── KKI
│       ├── NYU
│       ├── NeuroIMAGE
│       ├── OHSU
│       ├── Peking_1
│       ├── Pittsburgh
│       └── templates
├── Train
│   ├── RS
│   │   └── KKI
│   └── TC
│       ├── KKI
│       ├── NYU
│       ├── NeuroIMAGE
│       ├── OHSU
│       ├── Peking
│       ├── Peking_1
│       ├── Peking_2
│       ├── Peking_3
│       ├── Pittsburgh
│       ├── WashU
│       └── templates
└── other
```

26 directories

## Introduction

Landing conditions at Rankin Inlet (CYRT) are often unpredictable especially in the winter. When I wa working at Meliadine Gold Mine 25km away from Rankin Inlet we would be flown from Montreal to Rankin Inlet. Flights will be often delayed or postponed due to weather conditions. I had the idea to create a model to better predict if there would be safe landing conditions for when the plane arrived so less flights will need to be delayed or postponed. 

The model aims to predict if the landing conditions at Rankin Inlet will be favourable for landing 6 hours in advance. 6 hours is the approximate flight time from Montreal to Rankin Inlet. That way we can predict if a plane takes off now if the weather conditions will allow for a safe landing. 

## The Data
The European Centre for Medium-Range Weather Forecasts (ECMWF) provides historical gridded 2D weather data in GRIB format, with each grid cell representing a point estimate of weather conditions. For this project, I used: 

 t2m → 2 metre temperature 
 u10 → 10 metre U-component of wind 
 v10 → 10 metre V-component of wind 
 msl → Mean sea level pressure 

 ECMWF also provides current weather conditions, which I used to generate real-time predictions. 

 Additionally, metar-taf.com supplied historical METAR data, which I used to determine whether it was safe for a plane to land. This binary outcome served as the model’s response variable.

## The Model
Predictions were generated using a 2D convolutional neural network with four input channels, each representing a different weather variable. 

The model was trained on hourly data from 2022, validated on 2023 data, and tested on 2024 data. It achieved an AUC of 0.87 on the test set.

## Architecture

### Data Sources
| ECMWF API | METAR-TAF.com |
|:---------:|:-------------:|
| <img src="assets/ecmwfr.png" width="80"/> | <img src="assets/metar-taf.png" width="80"/> |

### Azure Cloud Architecture
| Azure Functions | Azure Data Lake Storage | Container App Jobs | Azure SQL |
|:---------------:|:-----------------------:|:------------------:|:---------:|
| <img src="assets/azure_functions.webp" width="80"/> | <img src="assets/adls.webp" width="80"/> | <img src="assets/containerappjobs.png" width="80"/> | <img src="assets/azure_sql.png" width="80"/> |
| Hourly Trigger | Raw Data Storage | Model Inference | Predictions Storage |

### Model Development
| PyTorch |
|:-------:|
| <img src="assets/pytorch.png" width="80"/> |

### Visualization
| Looker Dashboard |
|:----------------:|
| <img src="assets/looker.png" width="80"/> |

### Data Flow
```mermaid
flowchart TB
 subgraph Sources["External Data Sources"]
    direction LR
        ECMWF["ECMWF API"]
        METAR["METAR-TAF.com"]
  end
 subgraph Training["Model Development"]
        PyTorch["PyTorch Training"]
  end
 subgraph Azure["Azure Cloud"]
        Func["Azure Function App<br>(Hourly Trigger)"]
        ADLS[("Azure Data Lake Storage")]
        ACA["Azure Container App Job<br>(Inference)"]
        SQL[("Azure SQL Database")]
  end
 subgraph Output["Visualization"]
        Looker["Looker Dashboard"]
  end
    ECMWF -- Fetch Data --> Func
    METAR -- Fetch Data --> Func
    Func -- Ingest Raw Data --> ADLS
    PyTorch -. Deploy Model Artifact .-> ACA
    ADLS -- Read Batch Data --> ACA
    ACA -- Store Predictions --> SQL
    SQL -- Query Results --> Looker

     ECMWF:::external
     METAR:::external
     PyTorch:::process
     Func:::process
     ADLS:::storage
     ACA:::process
     SQL:::storage
     Looker:::external
    classDef storage fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef process fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef external fill:#fff3e0,stroke:#e65100,stroke-width:2px
```

![cyrt-safe-to-land](https://github.com/user-attachments/assets/98e0a847-ffa6-4213-a29e-cbe8cab5abf4)

To deploy use command: docker buildx build --platform linux/amd64 -t willseff/cyrt-inference:1.4 --push .

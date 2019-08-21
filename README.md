## Introduction

This is the source code and additional visualization examples of our Radial-GCN, Radial Graph Convolutional Network for Visual Question Generation.

1) Different from the existing approaches that typically treat the VQG task as a reversed VQA task, we propose a novel answer-centric approach for the VQG task, which effectively models the associations between the answer and its relevant image regions.
2)  To our best knowledge, we are the first to apply GCN model for the VQG task and devise a new radial graph structure with graphic attention for superior question generation performance and interpretable model behavior. 
3)  We conduct comprehensive experiments on three benchmark datasets to verify the advantage of our proposed method on generating meaningful questions on the VQG task and boosting the existing VQA methods on the challenging zero-shot VQA task.

![framework](https://github.com/submitwithanonymous/ICCV2019/raw/master/fig/framwork_new_.png)

<br>

## Code Structure

```
├── Radial-GCN/
|   ├── run_vqg.py          /* The main run files
|   ├── layer_vqg.py        /* Files for the model layer and structure (GCN, VQG)
|   ├── dataset_vqg.py      /* Files for construct vqg dataset
|   ├── utils.py            /* Files for tools
|   ├── main.py             /* Files for caption evaluation
|   ├── supp_questions      /* Files for generate questions for supplementary dataset for zero shot VQA
|   ├── draw_*.py           /* Files for drawing and visualisation
|   ├── readme.md
│   ├── ZS_VQA/
| 	├── data/                 /* Data file for zs_vqa
│   ├── data/                     /* Data files for training vqg
|	├── tools/                /* The modified file from bottom-up attention
|	├── process_image_vqg.py  /* Files for preprocess image
|	├── preprocess_text.py    /* Files for preprocess text
```

<br>

## Results




<table class="tg" align="center" border="1">
  <tr>
    <th class="tg-uys7" rowspan="2">Method</th>
    <th class="tg-uys7" colspan="5">VQA2</th>
    <th class="tg-c3ow" colspan="5">Visual7W</th>
  </tr>
  <tr>
    <th class="tg-c3ow">BLEU-1</td>
    <th class="tg-c3ow">BLEU-4</td>
    <th class="tg-c3ow">METEOR</td>
    <th class="tg-c3ow">CIDEr</td>
    <th class="tg-c3ow">ROUGE-L</td>
    <th class="tg-c3ow">BLEU-1</td>
    <th class="tg-c3ow">BLEU-4</td>
    <th class="tg-c3ow">METEOR</td>
    <th class="tg-c3ow">CIDEr</td>
    <th class="tg-c3ow">ROUGE-L</td>
  </tr>
  <tr>
    <td class="tg-c3ow">LSTM (Baseline)</td>
    <td class="tg-c3ow">0.381</td>
    <td class="tg-c3ow">0.152</td>
    <td class="tg-c3ow">0.198</td>
    <td class="tg-c3ow">1.32</td>
    <td class="tg-c3ow">0.471</td>
    <td class="tg-c3ow">0.447</td>
    <td class="tg-c3ow">0.202</td>
    <td class="tg-c3ow">0.192</td>
    <td class="tg-c3ow">1.13</td>
    <td class="tg-c3ow">0.468</td>
  </tr>
  <tr>
    <td class="tg-c3ow">LSTM-AN (Baseline)</td>
    <td class="tg-c3ow">0.492</td>
    <td class="tg-c3ow">0.228</td>
    <td class="tg-c3ow">0.243</td>
    <td class="tg-c3ow">1.62</td>
    <td class="tg-c3ow">0.526</td>
    <td class="tg-c3ow">0.463</td>
    <td class="tg-c3ow">0.219</td>
    <td class="tg-c3ow">0.229</td>
    <td class="tg-c3ow">1.34</td>
    <td class="tg-c3ow">0.501</td>
  </tr>
  <tr>
    <td class="tg-c3ow">SAT (ICML'15)</td>
    <td class="tg-c3ow">0.494</td>
    <td class="tg-c3ow">0.231</td>
    <td class="tg-c3ow">0.244</td>
    <td class="tg-c3ow">1.65</td>
    <td class="tg-c3ow">0.534</td>
    <td class="tg-c3ow">0.467</td>
    <td class="tg-c3ow">0.223</td>
    <td class="tg-c3ow">0.234</td>
    <td class="tg-c3ow">1.34</td>
    <td class="tg-c3ow">0.503</td>
  </tr>
  <tr>
    <td class="tg-c3ow">IVQA (CVPR'18)</td>
    <td class="tg-c3ow">0.502</td>
    <td class="tg-c3ow">0.239</td>
    <td class="tg-c3ow">0.257</td>
    <td class="tg-c3ow">1.84</td>
    <td class="tg-c3ow">0.553</td>
    <td class="tg-c3ow">0.472</td>
    <td class="tg-c3ow">0.227</td>
    <td class="tg-c3ow">0.237</td>
    <td class="tg-c3ow">1.36</td>
    <td class="tg-c3ow">0.508</td>
  </tr>
  <tr>
    <td class="tg-c3ow">iQAN (CVPR'18)</td>
    <td class="tg-c3ow">0.526</td>
    <td class="tg-c3ow">0.271</td>
    <td class="tg-c3ow">0.268</td>
    <td class="tg-c3ow">2.09</td>
    <td class="tg-c3ow">0.568</td>
    <td class="tg-c3ow">0.488</td>
    <td class="tg-c3ow">0.231</td>
    <td class="tg-c3ow">0.251</td>
    <td class="tg-c3ow">1.44</td>
    <td class="tg-c3ow">0.520</td>
  </tr>
  <tr>
    <td class="tg-7btt">Ours (w/o attention)</td>
    <td class="tg-c3ow">0.529</td>
    <td class="tg-c3ow">0.273</td>
    <td class="tg-c3ow">0.269</td>
    <td class="tg-c3ow">2.09</td>
    <td class="tg-c3ow">0.570</td>
    <td class="tg-c3ow">0.494</td>
    <td class="tg-c3ow">0.233</td>
    <td class="tg-c3ow">0.257</td>
    <td class="tg-c3ow">1.47</td>
    <td class="tg-c3ow">0.524</td>
  </tr>
  <tr>
    <td class="tg-7btt">Ours</td>
    <td class="tg-7btt">0.534</td>
    <td class="tg-7btt">0.279</td>
    <td class="tg-7btt">0.271</td>
    <td class="tg-7btt">2.10</td>
    <td class="tg-7btt">0.572</td>
    <td class="tg-7btt">0.501</td>
    <td class="tg-7btt">0.236</td>
    <td class="tg-7btt">0.259</td>
    <td class="tg-7btt">1.52</td>
    <td class="tg-7btt">0.527</td>
  </tr>
</table>


<br>

<table class="tg" align="center" border="1">
  <tr>
    <th class="tg-uys7" rowspan="2">Model</th>
    <th class="tg-uys7" colspan="2">VQA Model</th>
    <th class="tg-uys7" colspan="2">VQG Model</th>
    <th class="tg-c3ow" colspan="2">VQA val</th>
    <th class="tg-c3ow" colspan="2">Norm test</th>
    <th class="tg-c3ow" colspan="2">ZS-VQA test</th>
  </tr>
  <tr>
    <th class="tg-c3ow">Bottom-up</td>
    <th class="tg-c3ow">BAN</td>
    <th class="tg-c3ow">IVQA</td>
    <th class="tg-c3ow">Ours</td>
    <th class="tg-c3ow">Acc@1</td>
    <th class="tg-c3ow">Acc@Hum</td>
    <th class="tg-c3ow">Acc@1</td>
    <th class="tg-c3ow">Acc@Hum</td>
    <th class="tg-c3ow">Acc@1</td>
    <th class="tg-c3ow">Acc@Hum</td>
  </tr>
  <tr>
    <td class="tg-c3ow">1</td>
    <td class="tg-c3ow">√</td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow">59.6</td>
    <td class="tg-c3ow">66.6</td>
    <td class="tg-c3ow">48.8</td>
    <td class="tg-c3ow">56.9</td>
    <td class="tg-c3ow">0</td>
    <td class="tg-c3ow">0</td>
  </tr>
  <tr>
    <td class="tg-c3ow">2</td>
    <td class="tg-c3ow">√</td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow">√</td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow">59.0</td>
    <td class="tg-c3ow">66.1</td>
    <td class="tg-c3ow">48.3</td>
    <td class="tg-c3ow">56.0</td>
    <td class="tg-c3ow">29.2</td>
    <td class="tg-c3ow">39.4</td>
  </tr>
  <tr>
    <td class="tg-c3ow">3</td>
    <td class="tg-c3ow">√</td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow">√</td>
    <td class="tg-c3ow">59.1</td>
    <td class="tg-c3ow">66.3</td>
    <td class="tg-c3ow">48.3</td>
    <td class="tg-c3ow">56.2</td>
    <td class="tg-c3ow">30.1</td>
    <td class="tg-c3ow">40.4</td>
  </tr>
  <tr>
    <td class="tg-c3ow">4</td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow">√<br></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-7btt">60.6</td>
    <td class="tg-7btt">67.8</td>
    <td class="tg-7btt">49.8</td>
    <td class="tg-7btt">58.9</td>
    <td class="tg-c3ow">0</td>
    <td class="tg-c3ow">0</td>
  </tr>
  <tr>
    <td class="tg-c3ow">5</td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow">√</td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow">√</td>
    <td class="tg-c3ow">60.1</td>
    <td class="tg-c3ow">67.5</td>
    <td class="tg-c3ow">49.2</td>
    <td class="tg-c3ow">58.7</td>
    <td class="tg-7btt">30.7</td>
    <td class="tg-7btt">41.3</td>
  </tr>
</table>

<br>

## Visual Examples
More details can be refer to our **main text** and **supplementary**.
<br>
<br>
### View VQG Process  
![VQG Process](https://github.com/submitwithanonymous/ICCV2019/raw/master/fig/visual_new3.png)

<br>
<br>   

### View Question Distribution
![Distribution](https://github.com/submitwithanonymous/ICCV2019/raw/master/fig/tsne_vis.png)

<br>
<br>  

### View Supp. for ZS-VQA
![Supp](https://github.com/submitwithanonymous/ICCV2019/raw/master/fig/supp_q.png)

<br>
<br>  

### View More Examples
”**Q**”, “**A**” and “**Q***” denote the ground truth question, the given answer and generated question respectively.

![More Examples](https://github.com/submitwithanonymous/ICCV2019/raw/master/fig/visual3.png)


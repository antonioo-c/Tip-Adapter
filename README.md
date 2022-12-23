# Enhancing Tip-Adapter through knowledge fusion

Check out the original <code>README.md</code> in [Tip-Adapter.md](https://github.com/antonioo-c/Tip-Adapter/blob/main/Tip-Adapter.md), and the original code in [Tip-Adapter: Training-free Adaption of CLIP for Few-shot Classification](https://github.com/gaopengcuhk/Tip-Adapter).

## Updates

- [x] modified Tip-Adapter so that it supports [BLIP](https://github.com/salesforce/BLIP) in the cache model. Significant improvements can be seen.

## Results

<table>
<tr>
<th>Arch</th>
<th>Dataset</th>
<th>1</th>
<th>2</th>
<th>4</th>
<th>8</th>
<th>16</th>
</tr>
<tr>
<td>Tip-Adapter</td>
<td>ImageNet</td>
<td>60.70</td>
<td>60.96</td>
<td>60.98</td>
<td>61.45</td>
<td>62.01</td>
</tr>
<tr>
<td>Ours</td>
<td>ImageNet</td>
<td>60.84</td>
<td>61.42</td>
<td>62.37</td>
<td>63.92</td>
<td>66.40</td>
</tr>
<tr>
<td>Tip-Adapter-F</td>
<td>ImageNet</td>
<td>61.13</td>
<td>61.69</td>
<td>62.52</td>
<td>64.00</td>
<td>65.51</td>
</tr>
<tr>
</tr>
<td>Ours-F</td>
<td>ImageNet</td>
<td>61.80</td>
<td>62.74</td>
<td>65.07</td>
<td>68.09</td>
<td>71.22</td>
</tr>
<tr>
<tr>
<td>Tip-Adapter</td>
<td>Caltech101</td>
<td>87.18</td>
<td>88.44</td>
<td>89.39</td>
<td>89.83</td>
<td>90.18</td>
</tr>
<tr>
<td>Ours</td>
<td>Caltech101</td>
<td>89.33</td>
<td>90.95</td>
<td>93.23</td>
<td>94.69</td>
<td>95.17</td>
</tr>
<tr>
<td>Tip-Adapter-F</td>
<td>Caltech101</td>
<td>89.17</td>
<td>89.74</td>
<td>90.56</td>
<td>91.33</td>
<td>92.86</td>
</tr>
<tr>
</tr>
<td>Ours-F</td>
<td>Caltech101</td>
<td>90.79</td>
<td>91.89</td>
<td>94.48</td>
<td>95.58</td>
<td>96.75</td>
</tr>
<tr>




## Usage

Follow [Tip-Adapter.md](https://github.com/antonioo-c/Tip-Adapter/blob/main/Tip-Adapter.md) to prepare the datasets and python environments for the project. And then run

```bash
python main.py --config/datasets.yaml --use_blip. # Use blip model

python main.py --config/datasets.yaml . # No blip model, original setting.
```

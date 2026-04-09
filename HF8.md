### HF8 encoding details are specified as below. It has subnormal type and normal type. ###

<table style="text-align: center;">
  <tr>
    <th rowspan="2">Type</th>
    <th rowspan="2">Algo</th>
    <th colspan="4">Coding</th>
    <th colspan="3">Value</th>
  </tr>
  <tr>
    <td>S</td>
    <td>D</td>
    <td>E</td>
    <td>M</td>
    <td>S_v</td>
    <td>E_v</td>
    <td>M_v</td>
  </tr>
  <tr>
    <td>Subnormal</td>
    <td>S_v*2^(M_v-23)</td>
    <td>0 ~ 1</td>
    <td>0000</td>
    <td>-</td>
    <td>000~111</td>
    <td>±1</td>
    <td>-</td>
    <td>[0,7]</td>
  </tr>
  <tr>
    <td rowspan="5">Normal</td>
    <td rowspan="5">S_v*2^(E_v)*(1+M_v)</td>
    <td>0~1</td>
    <td>0001</td>
    <td>-</td>
    <td>000~111</td>
    <td>±1</td>
    <td>0</td>
    <td>[0/8, 7/8]</td>
  </tr>
  <tr>
    <td>0~1</td>
    <td>001</td>
    <td>0~1</td>
    <td>000~111</td>
    <td>±1</td>
    <td>±1</td>
    <td>[0/8, 7/8]</td>
  </tr>
    <tr>
    <td>0~1</td>
    <td>01</td>
    <td>00~11</td>
    <td>000~111</td>
    <td>±1</td>
    <td>±[2,3]</td>
    <td>[0/8, 7/8]</td>
  </tr>
    <tr>
    <td>0~1</td>
    <td>10</td>
    <td>000~111</td>
    <td>00~11</td>
    <td>±1</td>
    <td>±[4,7]</td>
    <td>[0/4, 3/4]</td>
  </tr>
    <tr>
    <td>0~1</td>
    <td>11</td>
    <td>0000~1111</td>
    <td>0~1</td>
    <td>±1</td>
    <td>±[8,15]</td>
    <td>[0/2, 1/2]</td>
  </tr>
</table>

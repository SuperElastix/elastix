/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
//
// \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
// Department of Radiology, Leiden, The Netherlands
//
// \note This work was funded by the Netherlands Organisation for
// Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
//
// OpenCL implementation of itk::IdentityTransform

//------------------------------------------------------------------------------
#ifdef DIM_1
float identity_transform_point_1d( const float point )
{
  return point;
}

#endif // DIM_1

//------------------------------------------------------------------------------
#ifdef DIM_2
float2 identity_transform_point_2d( const float2 point )
{
  return point;
}

#endif // DIM_2

//------------------------------------------------------------------------------
#ifdef DIM_3
float3 identity_transform_point_3d( const float3 point )
{
  return point;
}

#endif // DIM_3

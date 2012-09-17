/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

// OpenCL implementation of itk::TranslationTransformBase

// Definition of GPUTranslationTransformBase 1D/2D/3D
#ifdef DIM_1
typedef struct {
  float offset;
} GPUTranslationTransformBase1D;
#endif // DIM_1

#ifdef DIM_2
typedef struct {
  float2 offset;
} GPUTranslationTransformBase2D;
#endif // DIM_2

#ifdef DIM_3
typedef struct {
  float3 offset;
} GPUTranslationTransformBase3D;
#endif // DIM_3

//------------------------------------------------------------------------------
#ifdef DIM_1
float translation_transform_point_1d(const float point,
                                     __constant GPUTranslationTransformBase1D *transform_base)
{
  float tpoint = point + transform_base->offset;
  return tpoint;
}

#endif // DIM_1

//------------------------------------------------------------------------------
#ifdef DIM_2
float2 translation_transform_point_2d(const float2 point,
                                      __constant GPUTranslationTransformBase2D *transform_base)
{
  float2 tpoint = point + transform_base->offset;
  return tpoint;
}

#endif // DIM_2

//------------------------------------------------------------------------------
#ifdef DIM_3
float3 translation_transform_point_3d(const float3 point,
                                      __constant GPUTranslationTransformBase3D *transform_base)
{
  float3 tpoint = point + transform_base->offset;
  return tpoint;
}

#endif // DIM_3

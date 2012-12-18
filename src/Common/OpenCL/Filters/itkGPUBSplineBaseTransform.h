/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkGPUBSplineBaseTransform_h
#define __itkGPUBSplineBaseTransform_h

#include "itkGPUDataManager.h"
#include "itkGPUImage.h"
#include "itkGPUTransformBase.h"

namespace itk
{
/** \class GPUBSplineBaseTransform
* \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
* Department of Radiology, Leiden, The Netherlands
*
* This implementation was taken from elastix (http://elastix.isi.uu.nl/).
*
* \note This work was funded by the Netherlands Organisation for
* Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
*
*/
template< class TScalarType = float, unsigned int NDimensions = 3 >
class ITK_EXPORT GPUBSplineBaseTransform : public GPUTransformBase
{
public:
  /** Standard class typedefs. */
  typedef GPUBSplineBaseTransform Self;

  /** Run-time type information (and related methods). */
  itkTypeMacro( GPUBSplineBaseTransform, GPUTransformBase );

  typedef GPUImage< TScalarType, NDimensions >      GPUCoefficientImageType;
  typedef typename GPUCoefficientImageType::Pointer GPUCoefficientImagePointer;
  typedef typename GPUDataManager::Pointer          GPUDataManagerPointer;

  typedef FixedArray< GPUCoefficientImagePointer, NDimensions > GPUCoefficientImageArray;
  typedef FixedArray< GPUDataManagerPointer, NDimensions >      GPUCoefficientImageBaseArray;

  /**  */
  virtual bool IsBSplineTransform() const { return true; }

  /** Get the GPU array of coefficient images. */
  const GPUCoefficientImageArray GetGPUCoefficientImages() const
  {
    return this->m_GPUBSplineTransformCoefficientImages;
  }

  /** Get the GPU array of coefficient images bases. */
  const GPUCoefficientImageBaseArray GetGPUCoefficientImagesBases() const
  {
    return this->m_GPUBSplineTransformCoefficientImagesBase;
  }

protected:
  GPUBSplineBaseTransform() {}
  virtual ~GPUBSplineBaseTransform() {}

  GPUCoefficientImageArray     m_GPUBSplineTransformCoefficientImages;
  GPUCoefficientImageBaseArray m_GPUBSplineTransformCoefficientImagesBase;

private:
  GPUBSplineBaseTransform( const Self & other ); // purposely not implemented
  const Self & operator=( const Self & );        // purposely not implemented
};
} // end namespace itk

#endif /* __itkGPUBSplineBaseTransform_h */

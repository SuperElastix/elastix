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
#ifndef __itkGPUBSplineBaseTransform_h
#define __itkGPUBSplineBaseTransform_h

#include "itkGPUDataManager.h"
#include "itkGPUImage.h"
#include "itkGPUTransformBase.h"

namespace itk
{
/** Create a helper GPU Kernel class for GPUBSplineTransform */
itkGPUKernelClassMacro( GPUBSplineTransformKernel );

/** \class GPUBSplineBaseTransform
 * \brief GPU base class for the BSplineTransform.
 *
 * \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
 * Department of Radiology, Leiden, The Netherlands
 *
 * \note This work was funded by the Netherlands Organisation for
 * Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
 *
 * \ingroup GPUCommon
 */
template< typename TScalarType = float, unsigned int NDimensions = 3 >
class ITK_EXPORT GPUBSplineBaseTransform : public GPUTransformBase
{
public:

  /** Standard class typedefs. */
  typedef GPUBSplineBaseTransform Self;

  /** Run-time type information (and related methods). */
  itkTypeMacro( GPUBSplineBaseTransform, GPUTransformBase );

  /** Dimension of the domain space. */
  itkStaticConstMacro( SpaceDimension, unsigned int, NDimensions );

  /** Get the Spline Order, supports 0th - 3th order splines. The default
   * is a 3rd order spline. */
  itkGetConstMacro( SplineOrder, unsigned int );

  typedef GPUImage< TScalarType, NDimensions >      GPUCoefficientImageType;
  typedef typename GPUCoefficientImageType::Pointer GPUCoefficientImagePointer;
  typedef typename GPUDataManager::Pointer          GPUDataManagerPointer;

  typedef FixedArray< GPUCoefficientImagePointer, NDimensions > GPUCoefficientImageArray;
  typedef FixedArray< GPUDataManagerPointer, NDimensions >      GPUCoefficientImageBaseArray;

  /** Returns true, the transform is BSpline transform. */
  bool IsBSplineTransform( void ) const override { return true; }

  /** Get the GPU array of coefficient images. */
  const GPUCoefficientImageArray GetGPUCoefficientImages( void ) const;

  /** Get the GPU array of coefficient images bases. */
  const GPUCoefficientImageBaseArray GetGPUCoefficientImagesBases( void ) const;

protected:

  /** Sets the Spline Order, supports 0th - 3th order splines. The default
   * is a 3rd order spline. Should be protected method.
   * In ITK design the BSpline order is a template parameter of the
   * \see BSplineBaseTransform. We don't want the same definition
   * in GPUBSplineBaseTransform. Instead the derived class should
   * call SetSplineOrder() in the constructor. */
  virtual void SetSplineOrder( const unsigned int splineOrder );

  GPUBSplineBaseTransform();
  ~GPUBSplineBaseTransform() override {}

  /** Returns OpenCL \a source code for the transform.
   * Returns true if source code was combined, false otherwise. */
  bool GetSourceCode( std::string & source ) const override;

  GPUCoefficientImageArray     m_GPUBSplineTransformCoefficientImages;
  GPUCoefficientImageBaseArray m_GPUBSplineTransformCoefficientImagesBase;

private:

  GPUBSplineBaseTransform( const Self & other ); // purposely not implemented
  const Self & operator=( const Self & );        // purposely not implemented

  std::vector< std::string > m_Sources;

  // User specified spline order (3rd or cubic is the default)
  unsigned int m_SplineOrder;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkGPUBSplineBaseTransform.hxx"
#endif

#endif /* __itkGPUBSplineBaseTransform_h */

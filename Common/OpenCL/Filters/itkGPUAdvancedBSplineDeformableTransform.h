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
#ifndef itkGPUAdvancedBSplineDeformableTransform_h
#define itkGPUAdvancedBSplineDeformableTransform_h

#include "itkAdvancedBSplineDeformableTransform.h"
#include "itkGPUBSplineBaseTransform.h"

namespace itk
{
/** \class GPUAdvancedBSplineDeformableTransform
 * \brief GPU version of AdvancedBSplineDeformableTransform.
 *
 * \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
 * Department of Radiology, Leiden, The Netherlands
 *
 * \note This work was funded by the Netherlands Organisation for
 * Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
 *
 * \ingroup GPUCommon
 */
template <typename TScalarType = float,
          unsigned int NDimensions = 3,
          unsigned int VSplineOrder = 3,
          typename TParentTransform = AdvancedBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>>
class ITK_TEMPLATE_EXPORT GPUAdvancedBSplineDeformableTransform
  : public TParentTransform
  , public GPUBSplineBaseTransform<TScalarType, NDimensions>
{
public:
  /** Standard class typedefs. */
  using Self = GPUAdvancedBSplineDeformableTransform;
  using CPUSuperclass = TParentTransform;
  using GPUSuperclass = GPUBSplineBaseTransform<TScalarType, NDimensions>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;
  using ParametersType = typename CPUSuperclass::ParametersType;
  using ImagePointer = typename CPUSuperclass::ImagePointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Dimension of the domain space. */
  itkStaticConstMacro(SpaceDimension, unsigned int, NDimensions);

  /** Run-time type information (and related methods). */
  itkTypeMacro(GPUAdvancedBSplineDeformableTransform, TParentTransform);

  /** This method sets the parameters of the transform. */
  void
  SetParameters(const ParametersType & parameters) override;

  /** Set the array of coefficient images. */
  void
  SetCoefficientImages(ImagePointer images[]) override;

protected:
  GPUAdvancedBSplineDeformableTransform();
  ~GPUAdvancedBSplineDeformableTransform() override = default;

  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  void
  CopyCoefficientImagesToGPU();

private:
  GPUAdvancedBSplineDeformableTransform(const Self & other) = delete;
  const Self &
  operator=(const Self &) = delete;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkGPUAdvancedBSplineDeformableTransform.hxx"
#endif

#endif /* itkGPUAdvancedBSplineDeformableTransform_h */

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
#ifndef itkGPUAdvancedCombinationTransform_h
#define itkGPUAdvancedCombinationTransform_h

#include "itkAdvancedCombinationTransform.h"
#include "itkGPUCompositeTransformBase.h"

namespace itk
{
/** \class GPUAdvancedCombinationTransform
 * \brief GPU version of AdvancedCombinationTransform.
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
          typename TParentTransform = AdvancedCombinationTransform<TScalarType, NDimensions>>
class ITK_TEMPLATE_EXPORT GPUAdvancedCombinationTransform
  : public TParentTransform
  , public GPUCompositeTransformBase<TScalarType, NDimensions>
{
public:
  /** Standard class typedefs. */
  using Self = GPUAdvancedCombinationTransform;
  using CPUSuperclass = TParentTransform;
  using GPUSuperclass = GPUCompositeTransformBase<TScalarType, NDimensions>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(GPUAdvancedCombinationTransform, TParentTransform);

  /** Sub transform types. */
  using GPUTransformType = typename GPUSuperclass::TransformType;
  using TransformTypePointer = typename GPUSuperclass::TransformTypePointer;
  using TransformTypeConstPointer = typename GPUSuperclass::TransformTypeConstPointer;

  /** Get number of transforms in composite transform. */
  SizeValueType
  GetNumberOfTransforms() const override
  {
    return CPUSuperclass::GetNumberOfTransforms();
  }

  /** Get the Nth transform. */
  const TransformTypePointer
  GetNthTransform(SizeValueType n) const override
  {
    return CPUSuperclass::GetNthTransform(n);
  }

protected:
  GPUAdvancedCombinationTransform() = default;
  ~GPUAdvancedCombinationTransform() override = default;
  void
  PrintSelf(std::ostream & s, Indent indent) const override
  {
    CPUSuperclass::PrintSelf(s, indent);
  }

private:
  GPUAdvancedCombinationTransform(const Self & other) = delete;
  const Self &
  operator=(const Self &) = delete;
};

} // end namespace itk

#endif /* itkGPUAdvancedCombinationTransform_h */

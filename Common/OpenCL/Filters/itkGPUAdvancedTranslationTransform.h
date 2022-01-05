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
#ifndef itkGPUAdvancedTranslationTransform_h
#define itkGPUAdvancedTranslationTransform_h

#include "itkAdvancedTranslationTransform.h"
#include "itkGPUTranslationTransformBase.h"

namespace itk
{
/** \class GPUAdvancedTranslationTransform
 * \brief GPU version of AdvancedTranslationTransform.
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
          typename TParentTransform = AdvancedTranslationTransform<TScalarType, NDimensions>>
class ITK_TEMPLATE_EXPORT GPUAdvancedTranslationTransform
  : public TParentTransform
  , public GPUTranslationTransformBase<TScalarType, NDimensions>
{
public:
  /** Standard class typedefs. */
  using Self = GPUAdvancedTranslationTransform;
  using CPUSuperclass = TParentTransform;
  using GPUSuperclass = GPUTranslationTransformBase<TScalarType, NDimensions>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(GPUAdvancedTranslationTransform, CPUSuperclass);

  /** Typedefs from GPUSuperclass. */
  using CPUOutputVectorType = typename GPUSuperclass::CPUOutputVectorType;

  /** This method returns the CPU value of the offset of the TranslationTransform. */
  const CPUOutputVectorType &
  GetCPUOffset() const override
  {
    return this->GetOffset();
  }

protected:
  GPUAdvancedTranslationTransform() = default;
  ~GPUAdvancedTranslationTransform() override = default;

private:
  GPUAdvancedTranslationTransform(const Self & other) = delete;
  const Self &
  operator=(const Self &) = delete;
};

} // end namespace itk

#endif /* itkGPUAdvancedTranslationTransform_h */

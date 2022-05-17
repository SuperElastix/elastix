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
#ifndef itkGPUAdvancedEuler3DTransform_h
#define itkGPUAdvancedEuler3DTransform_h

#include "itkAdvancedEuler3DTransform.h"
#include "itkGPUMatrixOffsetTransformBase.h"

namespace itk
{
/** \class GPUAdvancedEuler3DTransform
 * \brief GPU version of AdvancedEuler3DTransform.
 *
 * \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
 * Department of Radiology, Leiden, The Netherlands
 *
 * \note This work was funded by the Netherlands Organisation for
 * Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
 *
 * \ingroup GPUCommon
 */
template <typename TScalarType = float, typename TParentTransform = AdvancedEuler3DTransform<TScalarType>>
class ITK_TEMPLATE_EXPORT GPUAdvancedEuler3DTransform
  : public TParentTransform
  , public GPUMatrixOffsetTransformBase<TScalarType, 3, 3>
{
public:
  /** Standard class typedefs. */
  using Self = GPUAdvancedEuler3DTransform;
  using CPUSuperclass = TParentTransform;
  using GPUSuperclass = GPUMatrixOffsetTransformBase<TScalarType, 3, 3>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(GPUAdvancedEuler3DTransform, CPUSuperclass);

  /** Typedefs from GPUSuperclass. */
  using CPUMatrixType = typename GPUSuperclass::CPUMatrixType;
  using CPUInverseMatrixType = typename GPUSuperclass::CPUInverseMatrixType;
  using CPUOutputVectorType = typename GPUSuperclass::CPUOutputVectorType;

  /** Get CPU matrix of an MatrixOffsetTransformBase. */
  const CPUMatrixType &
  GetCPUMatrix() const override
  {
    return this->GetMatrix();
  }

  /** Get CPU offset of an MatrixOffsetTransformBase. */
  const CPUOutputVectorType &
  GetCPUOffset() const override
  {
    return this->GetOffset();
  }

protected:
  GPUAdvancedEuler3DTransform() = default;
  ~GPUAdvancedEuler3DTransform() override = default;

private:
  GPUAdvancedEuler3DTransform(const Self & other) = delete;
  const Self &
  operator=(const Self &) = delete;
};

} // end namespace itk

#endif /* itkGPUAdvancedEuler3DTransform_h */

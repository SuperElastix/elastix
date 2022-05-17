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
#ifndef itkGPUSimilarity3DTransform_h
#define itkGPUSimilarity3DTransform_h

#include "itkSimilarity3DTransform.h"
#include "itkGPUMatrixOffsetTransformBase.h"

namespace itk
{
/** \class GPUSimilarity3DTransform
 * \brief GPU version of Similarity3DTransform.
 *
 * \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
 * Department of Radiology, Leiden, The Netherlands
 *
 * \note This work was funded by the Netherlands Organisation for
 * Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
 *
 * \ingroup GPUCommon
 */
template <typename TScalarType = float, typename TParentTransform = Similarity3DTransform<TScalarType>>
class ITK_TEMPLATE_EXPORT GPUSimilarity3DTransform
  : public TParentTransform
  , public GPUMatrixOffsetTransformBase<TScalarType, 3, 3>
{
public:
  /** Standard class typedefs. */
  using Self = GPUSimilarity3DTransform;
  using CPUSuperclass = TParentTransform;
  using GPUSuperclass = GPUMatrixOffsetTransformBase<TScalarType, 3, 3>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(GPUSimilarity3DTransform, CPUSuperclass);

  /** Typedefs from GPUSuperclass. */
  using CPUMatrixType = typename GPUSuperclass::CPUMatrixType;
  using CPUInverseMatrixType = typename GPUSuperclass::CPUInverseMatrixType;
  using CPUOutputVectorType = typename GPUSuperclass::CPUOutputVectorType;

  /** Get CPU matrix of an MatrixOffsetTransformBase. */
  virtual const CPUMatrixType &
  GetCPUMatrix() const
  {
    return this->GetMatrix();
  }

  /** Get CPU offset of an MatrixOffsetTransformBase. */
  virtual const CPUOutputVectorType &
  GetCPUOffset() const
  {
    return this->GetOffset();
  }

protected:
  GPUSimilarity3DTransform() {}

  virtual ~GPUSimilarity3DTransform() {}

private:
  GPUSimilarity3DTransform(const Self & other) = delete;
  const Self &
  operator=(const Self &) = delete;
};

} // end namespace itk

#endif /* itkGPUSimilarity3DTransform_h */

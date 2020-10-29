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
#ifndef itkGPUAdvancedEuler2DTransform_h
#define itkGPUAdvancedEuler2DTransform_h

#include "itkAdvancedRigid2DTransform.h"
#include "itkGPUMatrixOffsetTransformBase.h"

namespace itk
{
/** \class GPUAdvancedEuler2DTransform
 * \brief GPU version of AdvancedEuler2DTransform.
 *
 * \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
 * Department of Radiology, Leiden, The Netherlands
 *
 * \note This work was funded by the Netherlands Organisation for
 * Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
 *
 * \ingroup GPUCommon
 */
template <typename TScalarType = float, typename TParentTransform = AdvancedRigid2DTransform<TScalarType>>
class GPUAdvancedEuler2DTransform
  : public TParentTransform
  , public GPUMatrixOffsetTransformBase<TScalarType, 2, 2>
{
public:
  /** Standard class typedefs. */
  typedef GPUAdvancedEuler2DTransform                     Self;
  typedef TParentTransform                                CPUSuperclass;
  typedef GPUMatrixOffsetTransformBase<TScalarType, 2, 2> GPUSuperclass;
  typedef SmartPointer<Self>                              Pointer;
  typedef SmartPointer<const Self>                        ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(GPUAdvancedEuler2DTransform, CPUSuperclass);

  /** Typedefs from GPUSuperclass. */
  typedef typename GPUSuperclass::CPUMatrixType        CPUMatrixType;
  typedef typename GPUSuperclass::CPUInverseMatrixType CPUInverseMatrixType;
  typedef typename GPUSuperclass::CPUOutputVectorType  CPUOutputVectorType;

  /** Get CPU matrix of an MatrixOffsetTransformBase. */
  virtual const CPUMatrixType &
  GetCPUMatrix(void) const
  {
    return this->GetMatrix();
  }

  /** Get CPU inverse matrix of an MatrixOffsetTransformBase. */
  virtual const CPUInverseMatrixType &
  GetCPUInverseMatrix(void) const
  {
    return this->GetInverseMatrix();
  }

  /** Get CPU offset of an MatrixOffsetTransformBase. */
  virtual const CPUOutputVectorType &
  GetCPUOffset(void) const
  {
    return this->GetOffset();
  }

protected:
  GPUAdvancedEuler2DTransform() {}
  virtual ~GPUAdvancedEuler2DTransform() {}

private:
  GPUAdvancedEuler2DTransform(const Self & other); // purposely not implemented
  const Self &
  operator=(const Self &); // purposely not implemented
};

} // end namespace itk

#endif /* itkGPUAdvancedEuler2DTransform_h */

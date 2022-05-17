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
#ifndef itkGPUMatrixOffsetTransformBase_h
#define itkGPUMatrixOffsetTransformBase_h

#include "itkGPUTransformBase.h"
#include "itkMatrix.h"

namespace itk
{
/** Create a helper GPU Kernel class for itkGPUMatrixOffsetTransformBase */
itkGPUKernelClassMacro(GPUMatrixOffsetTransformBaseKernel);

/** \class GPUMatrixOffsetTransformBase
 * \brief Base version of the GPU MatrixOffsetTransform.
 *
 * \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
 * Department of Radiology, Leiden, The Netherlands
 *
 * \note This work was funded by the Netherlands Organisation for
 * Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
 *
 * \ingroup GPUCommon
 */
template <typename TScalarType = float,      // Data type for scalars
          unsigned int NInputDimensions = 3, // Number of dimensions in the input space
          unsigned int NOutputDimensions = 3>
// Number of dimensions in the output space
class ITK_EXPORT GPUMatrixOffsetTransformBase : public GPUTransformBase
{
public:
  /** Standard typedefs   */
  using Self = GPUMatrixOffsetTransformBase;
  using GPUSuperclass = GPUTransformBase;

  /** Run-time type information (and related methods). */
  itkTypeMacro(GPUMatrixOffsetTransformBase, GPUSuperclass);

  /**  */
  bool
  IsMatrixOffsetTransform() const override
  {
    return true;
  }

  /** Type of the scalar representing coordinate and vector elements. */
  using ScalarType = TScalarType;

  /** Dimension of the domain space. */
  itkStaticConstMacro(InputSpaceDimension, unsigned int, NInputDimensions);
  itkStaticConstMacro(OutputSpaceDimension, unsigned int, NOutputDimensions);
  itkStaticConstMacro(ParametersDimension, unsigned int, NOutputDimensions *(NInputDimensions + 1));

  /** Standard matrix type for this class */
  using CPUMatrixType = Matrix<TScalarType, Self::OutputSpaceDimension, Self::InputSpaceDimension>;
  using CPUInverseMatrixType = Matrix<TScalarType, Self::InputSpaceDimension, Self::OutputSpaceDimension>;
  using CPUOutputVectorType = Vector<TScalarType, Self::OutputSpaceDimension>;

  /** Get CPU matrix of an MatrixOffsetTransformBase. */
  virtual const CPUMatrixType &
  GetCPUMatrix() const = 0;

  /** Get CPU offset of an MatrixOffsetTransformBase. */
  virtual const CPUOutputVectorType &
  GetCPUOffset() const = 0;

protected:
  GPUMatrixOffsetTransformBase();
  ~GPUMatrixOffsetTransformBase() override = default;

  /** Returns OpenCL \a source code for the transform.
   * Returns true if source code was combined, false otherwise. */
  bool
  GetSourceCode(std::string & source) const override;

  /** Returns data manager that stores all settings for the transform. */
  GPUDataManager::Pointer
  GetParametersDataManager() const override;

private:
  GPUMatrixOffsetTransformBase(const Self & other) = delete;
  const Self &
  operator=(const Self &) = delete;

  std::vector<std::string> m_Sources;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkGPUMatrixOffsetTransformBase.hxx"
#endif

#endif /* itkGPUMatrixOffsetTransformBase_h */

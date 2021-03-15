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
#ifndef itkGPUTranslationTransformBase_h
#define itkGPUTranslationTransformBase_h

#include "itkGPUTransformBase.h"

namespace itk
{
/** Create a helper GPU Kernel class for itkGPUTranslationTransformBase */
itkGPUKernelClassMacro(GPUTranslationTransformBaseKernel);

/** \class GPUTranslationTransformBase
 * \brief Base class for all GPU translation transforms.
 *
 * \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
 * Department of Radiology, Leiden, The Netherlands
 *
 * \note This work was funded by the Netherlands Organisation for
 * Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
 *
 * \ingroup GPUCommon
 */
template <typename TScalarType = float, // Data type for scalars
          unsigned int NDimensions = 3>
class ITK_EXPORT GPUTranslationTransformBase : public GPUTransformBase
{
public:
  /** Standard typedefs   */
  typedef GPUTranslationTransformBase Self;
  typedef GPUTransformBase            GPUSuperclass;

  /** Run-time type information (and related methods). */
  itkTypeMacro(GPUTranslationTransformBase, GPUSuperclass);

  /** Returns true, the transform is translation transform. */
  bool
  IsTranslationTransform(void) const override
  {
    return true;
  }

  /** Type of the scalar representing coordinate and vector elements. */
  typedef TScalarType ScalarType;

  /** Dimension of the domain space. */
  itkStaticConstMacro(SpaceDimension, unsigned int, NDimensions);
  itkStaticConstMacro(ParametersDimension, unsigned int, NDimensions);

  /** Standard vector type for this class. */
  typedef Vector<TScalarType, NDimensions> CPUOutputVectorType;

  /** This method returns the CPU value of the offset of the TranslationTransform. */
  virtual const CPUOutputVectorType &
  GetCPUOffset(void) const = 0;

protected:
  GPUTranslationTransformBase();
  ~GPUTranslationTransformBase() override = default;

  /** Returns OpenCL \a source code for the transform.
   * Returns true if source code was combined, false otherwise. */
  bool
  GetSourceCode(std::string & source) const override;

  /** Returns data manager that stores all settings for the transform. */
  GPUDataManager::Pointer
  GetParametersDataManager(void) const override;

private:
  GPUTranslationTransformBase(const Self & other) = delete;
  const Self &
  operator=(const Self &) = delete;

  std::vector<std::string> m_Sources;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkGPUTranslationTransformBase.hxx"
#endif

#endif /* itkGPUTranslationTransformBase_h */

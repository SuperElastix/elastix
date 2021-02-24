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
#ifndef itkGPUIdentityTransform_h
#define itkGPUIdentityTransform_h

#include "itkIdentityTransform.h"
#include "itkVersion.h"

#include "itkGPUTransformBase.h"

namespace itk
{
/** Create a helper GPU Kernel class for GPUIdentityTransform */
itkGPUKernelClassMacro(GPUIdentityTransformKernel);

/** \class GPUIdentityTransform
 * \brief GPU version of IdentityTransform.
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
          typename TParentTransform = IdentityTransform<TScalarType, NDimensions>>
class ITK_TEMPLATE_EXPORT GPUIdentityTransform
  : public TParentTransform
  , public GPUTransformBase
{
public:
  /** Standard class typedefs. */
  typedef GPUIdentityTransform     Self;
  typedef TParentTransform         CPUSuperclass;
  typedef GPUTransformBase         GPUSuperclass;
  typedef SmartPointer<Self>       Pointer;
  typedef SmartPointer<const Self> ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(GPUIdentityTransform, TParentTransform);

  /** Returns true if the derived transform is identity transform,
   * false otherwise. */
  bool
  IsIdentityTransform(void) const override
  {
    return true;
  }

protected:
  GPUIdentityTransform();
  ~GPUIdentityTransform() override = default;
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  /** Returns OpenCL \a source code for the transform.
   * Returns true if source code was combined, false otherwise. */
  bool
  GetSourceCode(std::string & source) const override;

private:
  GPUIdentityTransform(const Self & other) = delete;
  const Self &
  operator=(const Self &) = delete;

  std::vector<std::string> m_Sources;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkGPUIdentityTransform.hxx"
#endif

#endif /* itkGPUIdentityTransform_h */

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
#ifndef itkGPUBSplineInterpolateImageFunction_h
#define itkGPUBSplineInterpolateImageFunction_h

#include "itkGPUInterpolateImageFunction.h"
#include "itkGPUDataManager.h"
#include "itkGPUImage.h"

#include "itkBSplineInterpolateImageFunction.h"
#include "itkVersion.h"

namespace itk
{
/** Create a helper GPU Kernel class for GPUBSplineInterpolateImageFunction */
itkGPUKernelClassMacro(GPUBSplineInterpolateImageFunctionKernel);

/** \class GPUBSplineInterpolateImageFunction
 * \brief GPU version of BSplineInterpolateImageFunction.
 *
 * \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
 * Department of Radiology, Leiden, The Netherlands
 *
 * \note This work was funded by the Netherlands Organisation for
 * Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
 *
 * \ingroup GPUCommon
 */
template <typename TInputImage, typename TCoordRep = float, typename TCoefficientType = float>
class ITK_EXPORT GPUBSplineInterpolateImageFunction
  : public GPUInterpolateImageFunction<TInputImage,
                                       TCoordRep,
                                       BSplineInterpolateImageFunction<TInputImage, TCoordRep, TCoefficientType>>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(GPUBSplineInterpolateImageFunction);

  /** Standard class typedefs. */
  using Self = GPUBSplineInterpolateImageFunction;
  using GPUSuperclass =
    GPUInterpolateImageFunction<TInputImage,
                                TCoordRep,
                                BSplineInterpolateImageFunction<TInputImage, TCoordRep, TCoefficientType>>;
  using CPUSuperclass =
    BSplineInterpolateImageFunction<TInputImage,
                                    TCoordRep,
                                    BSplineInterpolateImageFunction<TInputImage, TCoordRep, TCoefficientType>>;
  using Superclass = GPUSuperclass;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Run-time type information (and related methods). */
  itkTypeMacro(GPUBSplineInterpolateImageFunction, GPUSuperclass);

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** ImageDimension constants */
  itkStaticConstMacro(InputImageDimension, unsigned int, TInputImage::ImageDimension);

  using GPUCoefficientImageType = GPUImage<TCoefficientType, InputImageDimension>;
  using GPUCoefficientImagePointer = typename GPUCoefficientImageType::Pointer;
  using GPUDataManagerPointer = typename GPUDataManager::Pointer;

  /** Set the input image. This must be set by the user. */
  void
  SetInputImage(const TInputImage * inputData) override;

  /** Get the GPU coefficient image. */
  const GPUCoefficientImagePointer
  GetGPUCoefficients() const;

  /** Get the GPU coefficient image base. */
  const GPUDataManagerPointer
  GetGPUCoefficientsImageBase() const;

protected:
  GPUBSplineInterpolateImageFunction();
  ~GPUBSplineInterpolateImageFunction() override = default;
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  /** Returns OpenCL \a source code for the transform.
   * Returns true if source code was combined, false otherwise. */
  bool
  GetSourceCode(std::string & source) const override;

private:
  GPUCoefficientImagePointer m_GPUCoefficients;
  GPUDataManagerPointer      m_GPUCoefficientsImageBase;

  std::vector<std::string> m_Sources;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkGPUBSplineInterpolateImageFunction.hxx"
#endif

#endif /* itkGPUBSplineInterpolateImageFunction_h */

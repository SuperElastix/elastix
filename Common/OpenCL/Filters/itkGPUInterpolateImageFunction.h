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
#ifndef itkGPUInterpolateImageFunction_h
#define itkGPUInterpolateImageFunction_h

#include "itkInterpolateImageFunction.h"
#include "itkGPUInterpolatorBase.h"

namespace itk
{
/** \class GPUInterpolateImageFunction
 * \brief GPU version of InterpolateImageFunction.
 *
 * \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
 * Department of Radiology, Leiden, The Netherlands
 *
 * \note This work was funded by the Netherlands Organisation for
 * Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
 *
 * \ingroup GPUCommon
 */
template <typename TInputImage,
          typename TCoordRep = float,
          typename TParentInterpolateImageFunction = InterpolateImageFunction<TInputImage, TCoordRep>>
class ITK_TEMPLATE_EXPORT GPUInterpolateImageFunction
  : public TParentInterpolateImageFunction
  , public GPUInterpolatorBase
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(GPUInterpolateImageFunction);

  /** Standard class typedefs. */
  using Self = GPUInterpolateImageFunction;
  using CPUSuperclass = TParentInterpolateImageFunction;
  using GPUSuperclass = GPUInterpolatorBase;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Run-time type information (and related methods). */
  itkTypeMacro(GPUInterpolateImageFunction, TParentInterpolateImageFunction);

  /** ImageDimension constants */
  itkStaticConstMacro(InputImageDimension, unsigned int, TInputImage::ImageDimension);

  /** Superclass typedef support. */
  using InputImageType = typename CPUSuperclass::InputImageType;
  using ContinuousIndexType = typename CPUSuperclass::ContinuousIndexType;
  using CoordRepType = typename CPUSuperclass::CoordRepType;

protected:
  GPUInterpolateImageFunction();
  ~GPUInterpolateImageFunction() override = default;
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  /** Returns data manager that stores all settings for the transform. */
  GPUDataManager::Pointer
  GetParametersDataManager() const override;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkGPUInterpolateImageFunction.hxx"
#endif

#endif /* itkGPUInterpolateImageFunction_h */

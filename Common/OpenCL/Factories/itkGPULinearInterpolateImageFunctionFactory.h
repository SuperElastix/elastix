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
#ifndef itkGPULinearInterpolateImageFunctionFactory_h
#define itkGPULinearInterpolateImageFunctionFactory_h

#include "itkGPUObjectFactoryBase.h"
#include "itkGPULinearInterpolateImageFunction.h"

namespace itk
{
/** \class GPULinearInterpolateImageFunctionFactory2
 * \brief Object Factory implementation for GPULinearInterpolateImageFunction
 *
 * \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
 * Department of Radiology, Leiden, The Netherlands
 *
 * \note This work was funded by the Netherlands Organisation for
 * Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
 *
 */
template <typename TTypeList, typename NDimensions>
class ITK_TEMPLATE_EXPORT GPULinearInterpolateImageFunctionFactory2 : public GPUObjectFactoryBase<NDimensions>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(GPULinearInterpolateImageFunctionFactory2);

  using Self = GPULinearInterpolateImageFunctionFactory2;
  using Superclass = GPUObjectFactoryBase<NDimensions>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Return a descriptive string describing the factory. */
  const char *
  GetDescription() const override
  {
    return "A Factory for GPULinearInterpolateImageFunction";
  }

  /** Method for class instantiation. */
  itkFactorylessNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(GPULinearInterpolateImageFunctionFactory2, GPUObjectFactoryBase);

  /** Register one factory of this type. */
  static void
  RegisterOneFactory();

  /** Operator() to register override. */
  template <typename TType, unsigned int VImageDimension>
  void
  operator()()
  {
    // Image typedefs
    using InputImageType = Image<TType, VImageDimension>;
    using GPUInputImageType = GPUImage<TType, VImageDimension>;

    // Override default with the coordinate representation type as float
    this->RegisterOverride(typeid(LinearInterpolateImageFunction<InputImageType, float>).name(),
                           typeid(GPULinearInterpolateImageFunction<InputImageType, float>).name(),
                           "GPU LinearInterpolateImageFunction override with coord rep as float",
                           true,
                           CreateObjectFunction<GPULinearInterpolateImageFunction<InputImageType, float>>::New());

    // Override when itkGPUImage is first template argument
    // and the coordinate representation type as float
    this->RegisterOverride(typeid(LinearInterpolateImageFunction<GPUInputImageType, float>).name(),
                           typeid(GPULinearInterpolateImageFunction<GPUInputImageType, float>).name(),
                           "GPU LinearInterpolateImageFunction override for GPUImage with coord rep as float",
                           true,
                           CreateObjectFunction<GPULinearInterpolateImageFunction<GPUInputImageType, float>>::New());

    // Override default with and the coordinate representation type as double
    this->RegisterOverride(typeid(LinearInterpolateImageFunction<InputImageType, double>).name(),
                           typeid(GPULinearInterpolateImageFunction<InputImageType, double>).name(),
                           "GPU LinearInterpolateImageFunction override with coord rep as double",
                           true,
                           CreateObjectFunction<GPULinearInterpolateImageFunction<InputImageType, double>>::New());

    // Override when itkGPUImage is first template argument
    // and the coordinate representation type as double
    this->RegisterOverride(typeid(LinearInterpolateImageFunction<GPUInputImageType, double>).name(),
                           typeid(GPULinearInterpolateImageFunction<GPUInputImageType, double>).name(),
                           "GPU LinearInterpolateImageFunction override for GPUImage with coord rep as double",
                           true,
                           CreateObjectFunction<GPULinearInterpolateImageFunction<GPUInputImageType, double>>::New());
  }


protected:
  GPULinearInterpolateImageFunctionFactory2();
  ~GPULinearInterpolateImageFunctionFactory2() override = default;

  /** Register methods for 1D. */
  void
  Register1D() override;

  /** Register methods for 2D. */
  void
  Register2D() override;

  /** Register methods for 3D. */
  void
  Register3D() override;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkGPULinearInterpolateImageFunctionFactory.hxx"
#endif

#endif // end #ifndef itkGPULinearInterpolateImageFunctionFactory_h

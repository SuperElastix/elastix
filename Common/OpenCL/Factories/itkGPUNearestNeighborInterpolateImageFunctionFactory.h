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
#ifndef itkGPUNearestNeighborInterpolateImageFunctionFactory_h
#define itkGPUNearestNeighborInterpolateImageFunctionFactory_h

#include "itkGPUObjectFactoryBase.h"
#include "itkGPUNearestNeighborInterpolateImageFunction.h"

namespace itk
{
/** \class GPUNearestNeighborInterpolateImageFunctionFactory2
 * \brief Object Factory implementation for GPUNearestNeighborInterpolateImageFunction
 *
 * \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
 * Department of Radiology, Leiden, The Netherlands
 *
 * \note This work was funded by the Netherlands Organisation for
 * Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
 *
 */
template <typename TTypeList, typename NDimensions>
class ITK_TEMPLATE_EXPORT GPUNearestNeighborInterpolateImageFunctionFactory2 : public GPUObjectFactoryBase<NDimensions>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(GPUNearestNeighborInterpolateImageFunctionFactory2);

  using Self = GPUNearestNeighborInterpolateImageFunctionFactory2;
  using Superclass = GPUObjectFactoryBase<NDimensions>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Return a descriptive string describing the factory. */
  const char *
  GetDescription() const
  {
    return "A Factory for GPUNearestNeighborInterpolateImageFunction";
  }

  /** Method for class instantiation. */
  itkFactorylessNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(GPUNearestNeighborInterpolateImageFunctionFactory2, GPUObjectFactoryBase);

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
    this->RegisterOverride(
      typeid(NearestNeighborInterpolateImageFunction<InputImageType, float>).name(),
      typeid(GPUNearestNeighborInterpolateImageFunction<InputImageType, float>).name(),
      "GPU NearestNeighborInterpolateImageFunction override with coord rep as float",
      true,
      CreateObjectFunction<GPUNearestNeighborInterpolateImageFunction<InputImageType, float>>::New());

    // Override when itkGPUImage is first template argument
    // and the coordinate representation type as float
    this->RegisterOverride(
      typeid(NearestNeighborInterpolateImageFunction<GPUInputImageType, float>).name(),
      typeid(GPUNearestNeighborInterpolateImageFunction<GPUInputImageType, float>).name(),
      "GPU NearestNeighborInterpolateImageFunction override for GPUImage with coord rep as float",
      true,
      CreateObjectFunction<GPUNearestNeighborInterpolateImageFunction<GPUInputImageType, float>>::New());

    // Override default with and the coordinate representation type as double
    this->RegisterOverride(
      typeid(NearestNeighborInterpolateImageFunction<InputImageType, double>).name(),
      typeid(GPUNearestNeighborInterpolateImageFunction<InputImageType, double>).name(),
      "GPU NearestNeighborInterpolateImageFunction override with coord rep as double",
      true,
      CreateObjectFunction<GPUNearestNeighborInterpolateImageFunction<InputImageType, double>>::New());

    // Override when itkGPUImage is first template argument
    // and the coordinate representation type as double
    this->RegisterOverride(
      typeid(NearestNeighborInterpolateImageFunction<GPUInputImageType, double>).name(),
      typeid(GPUNearestNeighborInterpolateImageFunction<GPUInputImageType, double>).name(),
      "GPU NearestNeighborInterpolateImageFunction override for GPUImage with coord rep as double",
      true,
      CreateObjectFunction<GPUNearestNeighborInterpolateImageFunction<GPUInputImageType, double>>::New());
  }


protected:
  GPUNearestNeighborInterpolateImageFunctionFactory2();
  virtual ~GPUNearestNeighborInterpolateImageFunctionFactory2() {}

  /** Register methods for 1D. */
  virtual void
  Register1D();

  /** Register methods for 2D. */
  virtual void
  Register2D();

  /** Register methods for 3D. */
  virtual void
  Register3D();
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkGPUNearestNeighborInterpolateImageFunctionFactory.hxx"
#endif

#endif // end #ifndef itkGPUNearestNeighborInterpolateImageFunctionFactory_h

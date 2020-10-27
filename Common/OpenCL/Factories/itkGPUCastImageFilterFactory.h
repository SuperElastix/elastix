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
#ifndef __itkGPUCastImageFilterFactory_h
#define __itkGPUCastImageFilterFactory_h

#include "itkGPUObjectFactoryBase.h"
#include "itkGPUCastImageFilter.h"

namespace itk
{
/** \class GPUCastImageFilterFactory2
 * \brief Object Factory implementation for GPUCastImageFilter
 *
 * \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
 * Department of Radiology, Leiden, The Netherlands
 *
 * \note This work was funded by the Netherlands Organisation for
 * Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
 *
 */
template <typename TTypeListIn, typename TTypeListOut, typename NDimensions>
class GPUCastImageFilterFactory2 : public GPUObjectFactoryBase<NDimensions>
{
public:
  typedef GPUCastImageFilterFactory2        Self;
  typedef GPUObjectFactoryBase<NDimensions> Superclass;
  typedef SmartPointer<Self>                Pointer;
  typedef SmartPointer<const Self>          ConstPointer;

  /** Return a descriptive string describing the factory. */
  const char *
  GetDescription() const
  {
    return "A Factory for GPUCastImageFilter";
  }

  /** Method for class instantiation. */
  itkFactorylessNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(GPUCastImageFilterFactory2, GPUObjectFactoryBase);

  /** Register one factory of this type. */
  static void
  RegisterOneFactory();

  /** Operator() to register override. */
  template <typename TTypeIn, typename TTypeOut, unsigned int VImageDimension>
  void
  operator()(void)
  {
    // Image typedefs
    typedef Image<TTypeIn, VImageDimension>     InputImageType;
    typedef Image<TTypeOut, VImageDimension>    OutputImageType;
    typedef GPUImage<TTypeIn, VImageDimension>  GPUInputImageType;
    typedef GPUImage<TTypeOut, VImageDimension> GPUOutputImageType;

    // Override default
    this->RegisterOverride(typeid(CastImageFilter<InputImageType, OutputImageType>).name(),
                           typeid(GPUCastImageFilter<InputImageType, OutputImageType>).name(),
                           "GPU CastImageFilter override default",
                           true,
                           CreateObjectFunction<GPUCastImageFilter<InputImageType, OutputImageType>>::New());

    // Override when itkGPUImage is first template argument
    this->RegisterOverride(typeid(CastImageFilter<GPUInputImageType, OutputImageType>).name(),
                           typeid(GPUCastImageFilter<GPUInputImageType, OutputImageType>).name(),
                           "GPU CastImageFilter override GPUImage first",
                           true,
                           CreateObjectFunction<GPUCastImageFilter<GPUInputImageType, OutputImageType>>::New());

    // Override when itkGPUImage is second template argument
    this->RegisterOverride(typeid(CastImageFilter<InputImageType, GPUOutputImageType>).name(),
                           typeid(GPUCastImageFilter<InputImageType, GPUOutputImageType>).name(),
                           "GPU CastImageFilter override GPUImage second",
                           true,
                           CreateObjectFunction<GPUCastImageFilter<InputImageType, GPUOutputImageType>>::New());

    // Override when itkGPUImage is first and second template arguments
    this->RegisterOverride(typeid(CastImageFilter<GPUInputImageType, GPUOutputImageType>).name(),
                           typeid(GPUCastImageFilter<GPUInputImageType, GPUOutputImageType>).name(),
                           "GPU CastImageFilter override GPUImage first and second",
                           true,
                           CreateObjectFunction<GPUCastImageFilter<GPUInputImageType, GPUOutputImageType>>::New());
  }


protected:
  GPUCastImageFilterFactory2();
  virtual ~GPUCastImageFilterFactory2() {}

  /** Register methods for 1D. */
  virtual void
  Register1D();

  /** Register methods for 2D. */
  virtual void
  Register2D();

  /** Register methods for 3D. */
  virtual void
  Register3D();

private:
  GPUCastImageFilterFactory2(const Self &); // purposely not implemented
  void
  operator=(const Self &); // purposely not implemented
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkGPUCastImageFilterFactory.hxx"
#endif

#endif // end #ifndef __itkGPUCastImageFilterFactory_h

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
#ifndef __itkGPUResampleImageFilterFactory_h
#define __itkGPUResampleImageFilterFactory_h

#include "itkGPUObjectFactoryBase.h"
#include "itkGPUResampleImageFilter.h"

namespace itk
{
/** \class GPUResampleImageFilterFactory2
 * \brief Object Factory implementation for GPUResampleImageFilter
 *
 * \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
 * Department of Radiology, Leiden, The Netherlands
 *
 * \note This work was funded by the Netherlands Organisation for
 * Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
 *
 */
template< typename TTypeListIn, typename TTypeListOut, typename NDimentions >
class GPUResampleImageFilterFactory2 :
  public GPUObjectFactoryBase< NDimentions >
{
public:

  typedef GPUResampleImageFilterFactory2      Self;
  typedef GPUObjectFactoryBase< NDimentions > Superclass;
  typedef SmartPointer< Self >                Pointer;
  typedef SmartPointer< const Self >          ConstPointer;

  /** Return a descriptive string describing the factory. */
  const char * GetDescription() const { return "A Factory for GPUResampleImageFilter"; }

  /** Method for class instantiation. */
  itkFactorylessNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( GPUResampleImageFilterFactory2, GPUObjectFactoryBase );

  /** Register one factory of this type. */
  static void RegisterOneFactory();

  /** Operator() to register override. */
  template< typename TTypeIn, typename TTypeOut, unsigned int VImageDimension >
  void operator()( void )
  {
    // Image typedefs
    typedef Image< TTypeIn, VImageDimension >     InputImageType;
    typedef Image< TTypeOut, VImageDimension >    OutputImageType;
    typedef GPUImage< TTypeIn, VImageDimension >  GPUInputImageType;
    typedef GPUImage< TTypeOut, VImageDimension > GPUOutputImageType;

    // Override default
    this->RegisterOverride(
      typeid( ResampleImageFilter< InputImageType, OutputImageType, float > ).name(),
      typeid( GPUResampleImageFilter< InputImageType, OutputImageType, float > ).name(),
      "GPU ResampleImageFilter override default, interpolator float",
      true,
      CreateObjectFunction< GPUResampleImageFilter< InputImageType, OutputImageType, float > >::New()
      );

    // Override default
    // With interpolator precision type as double, on GPU as float
    this->RegisterOverride(
      typeid( ResampleImageFilter< InputImageType, OutputImageType, double > ).name(),
      typeid( GPUResampleImageFilter< InputImageType, OutputImageType, float > ).name(),
      "GPU ResampleImageFilter override default, interpolator double",
      true,
      CreateObjectFunction< GPUResampleImageFilter< InputImageType, OutputImageType, float > >::New()
      );

    // Override when itkGPUImage is first template argument
    this->RegisterOverride(
      typeid( ResampleImageFilter< GPUInputImageType, OutputImageType, float > ).name(),
      typeid( GPUResampleImageFilter< GPUInputImageType, OutputImageType, float > ).name(),
      "GPU ResampleImageFilter override GPUImage first, interpolator float",
      true,
      CreateObjectFunction< GPUResampleImageFilter< GPUInputImageType, OutputImageType, float > >::New()
      );

    // Override when itkGPUImage is first template argument
    // With interpolator precision type as double, on GPU as float
    this->RegisterOverride(
      typeid( ResampleImageFilter< GPUInputImageType, OutputImageType, double > ).name(),
      typeid( GPUResampleImageFilter< GPUInputImageType, OutputImageType, float > ).name(),
      "GPU ResampleImageFilter override GPUImage first, interpolator double",
      true,
      CreateObjectFunction< GPUResampleImageFilter< GPUInputImageType, OutputImageType, float > >::New()
      );

    // Override when itkGPUImage is second template argument
    this->RegisterOverride(
      typeid( ResampleImageFilter< InputImageType, GPUOutputImageType, float > ).name(),
      typeid( GPUResampleImageFilter< InputImageType, GPUOutputImageType, float > ).name(),
      "GPU ResampleImageFilter override GPUImage second, interpolator float",
      true,
      CreateObjectFunction< GPUResampleImageFilter< InputImageType, GPUOutputImageType, float > >::New()
      );

    // Override when itkGPUImage is second template argument
    // With interpolator precision type as double, on GPU as float
    this->RegisterOverride(
      typeid( ResampleImageFilter< InputImageType, GPUOutputImageType, double > ).name(),
      typeid( GPUResampleImageFilter< InputImageType, GPUOutputImageType, float > ).name(),
      "GPU ResampleImageFilter override GPUImage second, interpolator double",
      true,
      CreateObjectFunction< GPUResampleImageFilter< InputImageType, GPUOutputImageType, float > >::New()
      );

    // Override when itkGPUImage is first and second template arguments
    this->RegisterOverride(
      typeid( ResampleImageFilter< GPUInputImageType, GPUOutputImageType, float > ).name(),
      typeid( GPUResampleImageFilter< GPUInputImageType, GPUOutputImageType, float > ).name(),
      "GPU ResampleImageFilter override GPUImage first and second, interpolator float",
      true,
      CreateObjectFunction< GPUResampleImageFilter< GPUInputImageType, GPUOutputImageType, float > >::New()
      );

    // Override when itkGPUImage is first and second template arguments
    // With interpolator precision type as double, on GPU as float
    this->RegisterOverride(
      typeid( ResampleImageFilter< GPUInputImageType, GPUOutputImageType, double > ).name(),
      typeid( GPUResampleImageFilter< GPUInputImageType, GPUOutputImageType, float > ).name(),
      "GPU ResampleImageFilter override GPUImage first and second, interpolator double",
      true,
      CreateObjectFunction< GPUResampleImageFilter< GPUInputImageType, GPUOutputImageType, float > >::New()
      );
  }


protected:

  GPUResampleImageFilterFactory2();
  virtual ~GPUResampleImageFilterFactory2() {}

  /** Register methods for 1D. */
  virtual void Register1D();

  /** Register methods for 2D. */
  virtual void Register2D();

  /** Register methods for 3D. */
  virtual void Register3D();

private:

  GPUResampleImageFilterFactory2( const Self & ); // purposely not implemented
  void operator=( const Self & );                 // purposely not implemented

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkGPUResampleImageFilterFactory.hxx"
#endif

#endif // end #ifndef __itkGPUResampleImageFilterFactory_h

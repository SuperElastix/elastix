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
#ifndef __itkGPUShrinkImageFilterFactory_h
#define __itkGPUShrinkImageFilterFactory_h

#include "itkGPUObjectFactoryBase.h"
#include "itkGPUShrinkImageFilter.h"

namespace itk
{
/** \class GPUShrinkImageFilterFactory2
 * \brief Object Factory implementation for GPUShrinkImageFilter
 *
 * \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
 * Department of Radiology, Leiden, The Netherlands
 *
 * \note This work was funded by the Netherlands Organisation for
 * Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
 *
 */
template< typename TTypeListIn, typename TTypeListOut, typename NDimentions >
class GPUShrinkImageFilterFactory2 :
  public GPUObjectFactoryBase< NDimentions >
{
public:

  typedef GPUShrinkImageFilterFactory2        Self;
  typedef GPUObjectFactoryBase< NDimentions > Superclass;
  typedef SmartPointer< Self >                Pointer;
  typedef SmartPointer< const Self >          ConstPointer;

  /** Return a descriptive string describing the factory. */
  const char * GetDescription() const { return "A Factory for GPUShrinkImageFilter"; }

  /** Method for class instantiation. */
  itkFactorylessNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( GPUShrinkImageFilterFactory2, GPUObjectFactoryBase );

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
      typeid( ShrinkImageFilter< InputImageType, OutputImageType > ).name(),
      typeid( GPUShrinkImageFilter< InputImageType, OutputImageType > ).name(),
      "GPU ShrinkImageFilter override default",
      true,
      CreateObjectFunction< GPUShrinkImageFilter< InputImageType, OutputImageType > >::New()
      );

    // Override when itkGPUImage is first template argument
    this->RegisterOverride(
      typeid( ShrinkImageFilter< GPUInputImageType, OutputImageType > ).name(),
      typeid( GPUShrinkImageFilter< GPUInputImageType, OutputImageType > ).name(),
      "GPU ShrinkImageFilter override GPUImage first",
      true,
      CreateObjectFunction< GPUShrinkImageFilter< GPUInputImageType, OutputImageType > >::New()
      );

    // Override when itkGPUImage is second template argument
    this->RegisterOverride(
      typeid( ShrinkImageFilter< InputImageType, GPUOutputImageType > ).name(),
      typeid( GPUShrinkImageFilter< InputImageType, GPUOutputImageType > ).name(),
      "GPU ShrinkImageFilter override GPUImage second",
      true,
      CreateObjectFunction< GPUShrinkImageFilter< InputImageType, GPUOutputImageType > >::New()
      );

    // Override when itkGPUImage is first and second template arguments
    this->RegisterOverride(
      typeid( ShrinkImageFilter< GPUInputImageType, GPUOutputImageType > ).name(),
      typeid( GPUShrinkImageFilter< GPUInputImageType, GPUOutputImageType > ).name(),
      "GPU ShrinkImageFilter override GPUImage first and second",
      true,
      CreateObjectFunction< GPUShrinkImageFilter< GPUInputImageType, GPUOutputImageType > >::New()
      );
  }


protected:

  GPUShrinkImageFilterFactory2();
  virtual ~GPUShrinkImageFilterFactory2() {}

  /** Register methods for 1D. */
  virtual void Register1D();

  /** Register methods for 2D. */
  virtual void Register2D();

  /** Register methods for 3D. */
  virtual void Register3D();

private:

  GPUShrinkImageFilterFactory2( const Self & ); // purposely not implemented
  void operator=( const Self & );               // purposely not implemented

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkGPUShrinkImageFilterFactory.hxx"
#endif

#endif // end #ifndef __itkGPUShrinkImageFilterFactory_h

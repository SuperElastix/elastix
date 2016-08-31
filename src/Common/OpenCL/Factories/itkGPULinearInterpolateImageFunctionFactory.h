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
#ifndef __itkGPULinearInterpolateImageFunctionFactory_h
#define __itkGPULinearInterpolateImageFunctionFactory_h

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
template< typename TTypeList, typename NDimensions >
class GPULinearInterpolateImageFunctionFactory2 :
  public GPUObjectFactoryBase< NDimensions >
{
public:

  typedef GPULinearInterpolateImageFunctionFactory2 Self;
  typedef GPUObjectFactoryBase< NDimensions >       Superclass;
  typedef SmartPointer< Self >                      Pointer;
  typedef SmartPointer< const Self >                ConstPointer;

  /** Return a descriptive string describing the factory. */
  const char * GetDescription() const { return "A Factory for GPULinearInterpolateImageFunction"; }

  /** Method for class instantiation. */
  itkFactorylessNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( GPULinearInterpolateImageFunctionFactory2, GPUObjectFactoryBase );

  /** Register one factory of this type. */
  static void RegisterOneFactory();

  /** Operator() to register override. */
  template< typename TType, unsigned int VImageDimension >
  void operator()( void )
  {
    // Image typedefs
    typedef Image< TType, VImageDimension >    InputImageType;
    typedef GPUImage< TType, VImageDimension > GPUInputImageType;

    // Override default with the coordinate representation type as float
    this->RegisterOverride(
      typeid( LinearInterpolateImageFunction< InputImageType, float > ).name(),
      typeid( GPULinearInterpolateImageFunction< InputImageType, float > ).name(),
      "GPU LinearInterpolateImageFunction override with coord rep as float",
      true,
      CreateObjectFunction< GPULinearInterpolateImageFunction< InputImageType, float > >::New()
      );

    // Override when itkGPUImage is first template argument
    // and the coordinate representation type as float
    this->RegisterOverride(
      typeid( LinearInterpolateImageFunction< GPUInputImageType, float > ).name(),
      typeid( GPULinearInterpolateImageFunction< GPUInputImageType, float > ).name(),
      "GPU LinearInterpolateImageFunction override for GPUImage with coord rep as float",
      true,
      CreateObjectFunction< GPULinearInterpolateImageFunction< GPUInputImageType, float > >::New()
      );

    // Override default with and the coordinate representation type as double
    this->RegisterOverride(
      typeid( LinearInterpolateImageFunction< InputImageType, double > ).name(),
      typeid( GPULinearInterpolateImageFunction< InputImageType, double > ).name(),
      "GPU LinearInterpolateImageFunction override with coord rep as double",
      true,
      CreateObjectFunction< GPULinearInterpolateImageFunction< InputImageType, double > >::New()
      );

    // Override when itkGPUImage is first template argument
    // and the coordinate representation type as double
    this->RegisterOverride(
      typeid( LinearInterpolateImageFunction< GPUInputImageType, double > ).name(),
      typeid( GPULinearInterpolateImageFunction< GPUInputImageType, double > ).name(),
      "GPU LinearInterpolateImageFunction override for GPUImage with coord rep as double",
      true,
      CreateObjectFunction< GPULinearInterpolateImageFunction< GPUInputImageType, double > >::New()
      );
  }


protected:

  GPULinearInterpolateImageFunctionFactory2();
  virtual ~GPULinearInterpolateImageFunctionFactory2() {}

  /** Register methods for 1D. */
  virtual void Register1D();

  /** Register methods for 2D. */
  virtual void Register2D();

  /** Register methods for 3D. */
  virtual void Register3D();

private:

  GPULinearInterpolateImageFunctionFactory2( const Self & ); // purposely not implemented
  void operator=( const Self & );                            // purposely not implemented

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkGPULinearInterpolateImageFunctionFactory.hxx"
#endif

#endif // end #ifndef __itkGPULinearInterpolateImageFunctionFactory_h

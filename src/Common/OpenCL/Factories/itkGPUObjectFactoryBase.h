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
#ifndef __itkGPUObjectFactoryBase_h
#define __itkGPUObjectFactoryBase_h

#include "itkGPUSupportedImages.h"

// ITK includes
#include "itkVersion.h"
#include "itkObjectFactoryBase.h"

namespace itk
{
/** \class GPUObjectFactoryBase
 * \brief Base class for all GPU factory classes.
 *
 * \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
 * Department of Radiology, Leiden, The Netherlands
 *
 * \note This work was funded by the Netherlands Organisation for
 * Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
 *
 */
template< typename NDimentions >
class GPUObjectFactoryBase : public ObjectFactoryBase
{
public:

  typedef GPUObjectFactoryBase       Self;
  typedef ObjectFactoryBase          Superclass;
  typedef SmartPointer< Self >       Pointer;
  typedef SmartPointer< const Self > ConstPointer;

  /** Class methods used to interface with the registered factories. */
  virtual const char * GetITKSourceVersion() const { return ITK_SOURCE_VERSION; }

  /** Run-time type information (and related methods). */
  itkTypeMacro( GPUObjectFactoryBase, ObjectFactoryBase );

  /** Supported dimensions. */
  itkStaticConstMacro( Support1D, bool, NDimentions::Support1D );
  itkStaticConstMacro( Support2D, bool, NDimentions::Support2D );
  itkStaticConstMacro( Support3D, bool, NDimentions::Support3D );

  /** Main register method. This method is usually called by the derived
   * class in the constructor or after UnRegisterAllFactories() was called. */
  virtual void RegisterAll();

protected:

  GPUObjectFactoryBase() {}
  virtual ~GPUObjectFactoryBase() {}

  /** Register methods for 1D. */
  virtual void Register1D() {}

  /** Register methods for 2D. */
  virtual void Register2D() {}

  /** Register methods for 3D. */
  virtual void Register3D() {}

private:

  GPUObjectFactoryBase( const Self & ); // purposely not implemented
  void operator=( const Self & );       // purposely not implemented

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkGPUObjectFactoryBase.hxx"
#endif

#endif // end #ifndef __itkGPUObjectFactoryBase_h

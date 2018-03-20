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
#ifndef __itkMevisDicomTiffImageIOFactory_h
#define __itkMevisDicomTiffImageIOFactory_h

#include "itkObjectFactoryBase.h"
#include "itkImageIOBase.h"

namespace itk
{

/** \class MevisDicomTiffImageIOFactory
 * \brief Create instances of MevisDicomTiffImageIO objects using an object factory.
 */

class ITK_EXPORT MevisDicomTiffImageIOFactory : public ObjectFactoryBase
{
public:

  /** Standard class typedefs. */
  typedef MevisDicomTiffImageIOFactory Self;
  typedef ObjectFactoryBase            Superclass;
  typedef SmartPointer< Self >         Pointer;
  typedef SmartPointer< const Self >   ConstPointer;

  /** Class methods used to interface with the registered factories. */
  virtual const char * GetITKSourceVersion( void ) const;

  virtual const char * GetDescription( void ) const;

  /** Method for class instantiation. */
  itkFactorylessNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( MevisDicomTiffImageIOFactory, ObjectFactoryBase );

  /** Register one factory of this type  */
  static void RegisterOneFactory( void )
  {
    MevisDicomTiffImageIOFactory::Pointer metaFactory = MevisDicomTiffImageIOFactory::New();
    ObjectFactoryBase::RegisterFactory( metaFactory );
  }


protected:

  MevisDicomTiffImageIOFactory();
  ~MevisDicomTiffImageIOFactory();

private:

  MevisDicomTiffImageIOFactory( const Self & ); //purposely not implemented
  void operator=( const Self & );               //purposely not implemented

};

// end class MevisDicomTiffImageIOFactory

} // end namespace itk

#endif // end #ifndef __itkMevisDicomTiffImageIOFactory_h

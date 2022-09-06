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
#ifndef itkMevisDicomTiffImageIOFactory_h
#define itkMevisDicomTiffImageIOFactory_h

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
  ITK_DISALLOW_COPY_AND_MOVE(MevisDicomTiffImageIOFactory);

  /** Standard class typedefs. */
  using Self = MevisDicomTiffImageIOFactory;
  using Superclass = ObjectFactoryBase;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Class methods used to interface with the registered factories. */
  virtual const char *
  GetITKSourceVersion() const;

  virtual const char *
  GetDescription() const;

  /** Method for class instantiation. */
  itkFactorylessNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(MevisDicomTiffImageIOFactory, ObjectFactoryBase);

  /** Register one factory of this type  */
  static void
  RegisterOneFactory()
  {
    auto metaFactory = MevisDicomTiffImageIOFactory::New();
    ObjectFactoryBase::RegisterFactory(metaFactory);
  }


protected:
  MevisDicomTiffImageIOFactory();
  ~MevisDicomTiffImageIOFactory();
};

// end class MevisDicomTiffImageIOFactory

} // end namespace itk

#endif // end #ifndef itkMevisDicomTiffImageIOFactory_h

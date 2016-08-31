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

#ifndef __elxInstallFunctions_h
#define __elxInstallFunctions_h

#include "elxComponentDatabase.h"

namespace elastix
{

/**
 * \class InstallFunctions
 *
 * \brief A class with functions that are used to install elastix components
 *
 * In elastix the metric/transform/dimension/pixeltype etc. are all selected
 * at runtime. To make this possible, all components (metric/transform etc)
 * have to compiled for different dimension/pixeltype. The elx::ComponentDatabase
 * stores for each instance and each pixeltype/dimension a pointers to a function
 * that creates a component of the specific type. The InstallFunctions
 * class provides functions that aid in filling the elx::ComponentDatabase.
 * The functions are called when elastix is started. Do not do this
 * directly. Use the elxInstallMacro instead (see elxMacro.h).
 *
 * \sa ComponentDatabase
 * \ingroup Install
 */

template< class TAnyItkObject >
class InstallFunctions
{
public:

  /** Standard.*/
  typedef InstallFunctions Self;
  typedef TAnyItkObject    AnyItkObjectType;

  /** The baseclass of all objects that are returned by the Creator. */
  typedef ComponentDatabase::ObjectType    ObjectType;
  typedef ComponentDatabase::ObjectPointer ObjectPointer;

  /** The type of the index in the component database.
   * Each combination of pixeltype/dimension corresponds
   * a specific number, the index (unsigned int). */
  typedef ComponentDatabase::IndexType IndexType;

  /** The type of the key in the component database (=string) */
  typedef ComponentDatabase::ComponentDescriptionType ComponentDescriptionType;

  /** A wrap around the New() functions of itkObjects. */
  static ObjectPointer Creator( void )
  {
    return dynamic_cast< ObjectType * >( AnyItkObjectType::New().GetPointer() );
  }


  /** This function places the address of the New() function
   * of AnyItkObjectType in the ComponentDatabase. Returns 0 in
   * case of no errors.  */
  static int InstallComponent(
    const ComponentDescriptionType & name,
    IndexType i, ComponentDatabase * cdb )
  {
    return cdb->SetCreator( name, i, Self::Creator );
  }


};

} // end namespace elastix

#endif // end #ifndef __elxInstallFunctions_h

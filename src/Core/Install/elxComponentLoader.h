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

#ifndef __elxComponentLoader_h
#define __elxComponentLoader_h

#include "elxComponentDatabase.h"
#include "xoutmain.h"

namespace elastix
{

/**
* \class ComponentLoader
*
* \brief Determines which components (metrics, transforms, etc.) are available.
*
* This file defines the class elx::ComponentLoader. This class
* stores pointers to the New() functions of
* each component in the elx::ComponentDatabase.
*
* Each new component (a new metric for example should "make itself
* known" by calling the elxInstallMacro, which is defined in elxMacro.h.
*/

class ComponentLoader : public itk::Object
{
public:

  /** Standard ITK typedef's. */
  typedef ComponentLoader                 Self;
  typedef itk::Object                     Superclass;
  typedef itk::SmartPointer< Self >       Pointer;
  typedef itk::SmartPointer< const Self > ConstPointer;

  /** Standard ITK stuff. */
  itkNewMacro( Self );
  itkTypeMacro( ComponentLoader, Object );

  /** Typedef's. */
  typedef ComponentDatabase              ComponentDatabaseType;
  typedef ComponentDatabaseType::Pointer ComponentDatabasePointer;

  /** Set and get the ComponentDatabase. */
  itkSetObjectMacro( ComponentDatabase, ComponentDatabaseType );
  itkGetObjectMacro( ComponentDatabase, ComponentDatabaseType );

  /** Function to load components. The argv0 used to be useful
   * to find the program directory, but is not used anymore. */
  virtual int LoadComponents( const char * argv0 );

  /** Function to unload components. */
  virtual void UnloadComponents( void );

protected:

  /** Standard constructor and destructor. */
  ComponentLoader();
  virtual ~ComponentLoader();

  ComponentDatabasePointer m_ComponentDatabase;

  bool m_ImageTypeSupportInstalled;
  virtual int   InstallSupportedImageTypes( void );

private:

  /** Standard private (copy)constructor. */
  ComponentLoader( const Self & );  // purposely not implemented
  void operator=( const Self & );   // purposely not implemented

};

} //end namespace elastix

#endif // #ifndef __elxComponentLoader_h

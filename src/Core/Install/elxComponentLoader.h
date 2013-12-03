/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

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

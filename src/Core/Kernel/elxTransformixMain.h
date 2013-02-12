/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __elxTransformixMain_H_
#define __elxTransformixMain_H_

#include "elxElastixMain.h"

namespace elastix
{

/**
 * \class TransformixMain
 * \brief A class with all functionality to configure transformix.
 *
 * The TransformixMain class inherits from ElastixMain. We overwrite the Run()
 * -function. In the new Run() the Run()-function from the
 * ElastixTemplate-class is not called (as in elxElastixMain.cxx),
 * because this time we don't want to start a registration, but
 * just apply a transformation to an input image.
 *
 * \ingroup Kernel
 */

class TransformixMain : public ElastixMain
{
public:

  /** Standard itk. */
  typedef TransformixMain                Self;
  typedef ElastixMain                    Superclass;
  typedef itk::SmartPointer<Self>        Pointer;
  typedef itk::SmartPointer<const Self>  ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( TransformixMain, ElastixMain );

  /** Typedef's from Superclass. */

  /** typedef's from itk base Object. */
  typedef Superclass::ObjectType          ObjectType;
  typedef Superclass::ObjectPointer       ObjectPointer;
  typedef Superclass::DataObjectType      DataObjectType;
  typedef Superclass::DataObjectPointer   DataObjectPointer;

  /** Elastix components. */
  typedef Superclass::ElastixBaseType             ElastixBaseType;
  typedef Superclass::ConfigurationType           ConfigurationType;
  typedef Superclass::ArgumentMapType             ArgumentMapType;
  typedef Superclass::ConfigurationPointer        ConfigurationPointer;
  typedef Superclass::ObjectContainerType         ObjectContainerType;
  typedef Superclass::DataObjectContainerType     DataObjectContainerType;
  typedef Superclass::ObjectContainerPointer      ObjectContainerPointer;
  typedef Superclass::DataObjectContainerPointer  DataObjectContainerPointer;

  /** Typedefs for the database that holds pointers to New() functions.
   * Those functions are used to instantiate components, such as the metric etc.
   */
  typedef Superclass::ComponentDatabaseType       ComponentDatabaseType;
  typedef Superclass::ComponentDatabasePointer    ComponentDatabasePointer;
  typedef Superclass::PtrToCreator                PtrToCreator;
  typedef Superclass::ComponentDescriptionType    ComponentDescriptionType;
  typedef Superclass::PixelTypeDescriptionType    PixelTypeDescriptionType;
  typedef Superclass::ImageDimensionType          ImageDimensionType;
  typedef Superclass::DBIndexType                 DBIndexType;

  /** Typedef for class that populates a ComponentDatabase. */
  typedef Superclass::ComponentLoaderType         ComponentLoaderType;
  typedef Superclass::ComponentLoaderPointer      ComponentLoaderPointer;

  /** Typedef that is used in the elastix dll version */
  typedef Superclass::ParameterMapType            ParameterMapType;

  /** Overwrite Run() from base-class. */
  virtual int Run( void );

  /** Overwrite Run( argmap ) from superclass. Simply calls the superclass. */
  virtual int Run( ArgumentMapType & argmap );

  /** Get and Set input- and outputImage. */
  virtual void SetInputImageContainer(
    DataObjectContainerType * inputImageContainer );

protected:

  TransformixMain(){};
  virtual ~TransformixMain(){};

  /** InitDBIndex sets m_DBIndex to the value obtained
   * from the ComponentDatabase.
   */
  virtual int InitDBIndex( void );

private:

  TransformixMain( const Self& ); // purposely not implemented
  void operator=( const Self& );  // purposely not implemented

}; // end class TransformixMain


} // end namespace elastix


#endif // end #ifndef __elxTransformixMain_h

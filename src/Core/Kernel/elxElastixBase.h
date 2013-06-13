/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

/**
 * This file contains the declaration of the elx::ElastixBase class.
 * elx::ElastixTemplate<> inherits from this class. It is an abstract class,
 * since it contains pure virtual functions (which must be implemented
 * in ElastixTemplate<>).
 *
 * The Configuration object is stored in this class.
 */

#ifndef __elxElastixBase_h
#define __elxElastixBase_h

#include "elxBaseComponent.h"
#include "elxComponentDatabase.h"
#include "elxConfiguration.h"
#include "itkObject.h"
#include "itkDataObject.h"
#include "elxMacro.h"
#include "xoutmain.h"
#include "itkVectorContainer.h"
#include "itkImageFileReader.h"
#include "itkChangeInformationImageFilter.h"

#include <fstream>
#include <iomanip>

/** Like itkGet/SetObjectMacro, but in these macros the itkDebugMacro is
 * not called. Besides, they are not virtual, since
 * for now we do not need to override them somewhere.
 *
 * These macros are undef'd at the end of this file
 */
#define elxGetObjectMacro(_name,_type) \
  virtual _type * Get##_name (void) const \
  { \
    return this->m_##_name .GetPointer(); \
  }
//end elxGetObjectMacro

#define elxSetObjectMacro(_name,_type) \
  virtual void Set##_name (_type * _arg) \
  { \
    if ( this->m_##_name != _arg ) \
    { \
      this->m_##_name = _arg; \
      this->GetAsITKBaseType()->Modified(); \
    } \
  }
//end elxSetObjectMacro

/** defines for example: GetNumberOfMetrics() */
#define elxGetNumberOfMacro(_name) \
  virtual unsigned int GetNumberOf##_name##s(void) const \
  { \
    if ( this->Get##_name##Container() != 0 ) \
    { \
      return this->Get##_name##Container()->Size(); \
    } \
    return 0; \
  }
// end elxGetNumberOfMacro

namespace elastix
{
/**
 * \class ElastixBase
 * \brief This class creates an interface for elastix.
 *
 * The ElastixBase class creates an interface for elastix.
 * This is specified in ElastixTemplate, where all functions are defined.
 * Functionality that does not depend on the pixel type and the dimension
 * of the images to be registered, is defined in this class.
 *
 * The parameters used by this class are:
 * \parameter RandomSeed: Sets a global seed for the random generator.\n
 *   example: <tt>(RandomSeed 121212)</tt>\n
 *   It must be a positive integer number. Default: 121212.
 * \parameter DefaultOutputPrecision: Set the default precision of floating values in the output.
 *   Most importantly, it affects the output precision of the parameters in the transform parameter file.\n
 *   example: <tt>(DefaultOutputPrecision 6)</tt>\n
 *   Default value: 6.
 *
 * The command line arguments used by this class are:
 * \commandlinearg -f: mandatory argument for elastix with the file name of the fixed image. \n
 *    example: <tt>-f fixedImage.mhd</tt> \n
 * \commandlinearg -m: mandatory argument for elastix with the file name of the moving image. \n
 *    example: <tt>-m movingImage.mhd</tt> \n
 * \commandlinearg -out: mandatory argument for both elastix and transformix
 *    with the name of the directory that is going to contain everything that
 *    elastix or tranformix returns as output. \n
 *    example: <tt>-out outputdirectory</tt> \n
 * \commandlinearg -p: mandatory argument for elastix with the name of the parameter file. \n
 *    example: <tt>-p parameters.txt</tt> \n
 *    Multiple parameter files are allowed. It means that multiple registrations
 *    are runned in sequence, with the output of some registration as input
 *    to the next.
 * \commandlinearg -fMask: Optional argument for elastix with the file name of a mask for
 *    the fixed image. The mask image should contain of zeros and ones, zeros indicating
 *    pixels that are not used for the registration. \n
 *    example: <tt>-fMask fixedmask.mhd</tt> \n
 * \commandlinearg -mMask: Optional argument for elastix with the file name of a mask for
 *    the moving image. The mask image should contain of zeros and ones, zeros indicating
 *    pixels that are not used for the registration. \n
 *    example: <tt>-mMask movingmask.mhd</tt> \n
 * \commandlinearg -tp: mandatory argument for transformix with the name of
 *    the transform parameter file. \n
 *    example: <tt>-tp TransformParameters.txt</tt> \n
 *    In one such a transform parameter file a reference can be used to another
 *    transform parameter file, which is then used as an initial transform.
 * \commandlinearg -priority: optional argument for both elastix and transformix to
 *    specify the priority setting of this process. Choose one from {belownormal, high}. \n
 *    example: <tt>-priority high</tt> \n
 *    This argument is only valid for running under Windows. For Linux, run
 *    elastix with "nice".
 * \commandlinearg -threads: optional argument for both elastix and transformix to
 *    specify the maximum number of threads used by this process. Default: no maximum. \n
 *    example: <tt>-threads 2</tt> \n
 * \commandlinearg -in: optional argument for transformix with the file name of an input image. \n
 *    example: <tt>-in inputImage.mhd</tt> \n
 *    If this option is skipped, a deformation field of the transform will be generated.
 *
 * \ingroup Kernel
 */

class ElastixBase : public BaseComponent
{
public:

  /** Standard typedefs etc. */
  typedef ElastixBase       Self;
  typedef BaseComponent     Superclass;

  /** Typedefs used in this class. */
  typedef Configuration                       ConfigurationType;
  typedef ConfigurationType::Pointer          ConfigurationPointer;
  typedef itk::Object                         ObjectType; //for the components
  typedef ObjectType::Pointer                 ObjectPointer;
  typedef itk::DataObject                     DataObjectType; //for the images
  typedef DataObjectType::Pointer             DataObjectPointer;
  typedef itk::VectorContainer<
    unsigned int, ObjectPointer>              ObjectContainerType;
  typedef ObjectContainerType::Pointer        ObjectContainerPointer;
  typedef itk::VectorContainer<
    unsigned int, DataObjectPointer>          DataObjectContainerType;
  typedef DataObjectContainerType::Pointer    DataObjectContainerPointer;
  typedef itk::VectorContainer<
    unsigned int, std::string >               FileNameContainerType;
  typedef FileNameContainerType::Pointer      FileNameContainerPointer;

  /** Other typedef's. */
  typedef ComponentDatabase                   ComponentDatabaseType;
  typedef ComponentDatabaseType::Pointer      ComponentDatabasePointer;
  typedef ComponentDatabaseType::IndexType    DBIndexType;
  typedef std::vector<double>                 FlatDirectionCosinesType;

  /** Typedef that is used in the elastix dll version. */
  typedef itk::ParameterMapInterface::ParameterMapType  ParameterMapType;

  /** The itk class that ElastixTemplate is expected to inherit from
   * Of course ElastixTemplate also inherits from this class (ElastixBase).
   */
  typedef itk::Object      ITKBaseType;

  /** Cast to ITKBaseType. */
  virtual ITKBaseType * GetAsITKBaseType( void )
  {
    return dynamic_cast<ITKBaseType *>( this );
  }

  /** Set/Get the Configuration Object. */
  elxGetObjectMacro( Configuration, ConfigurationType );
  elxSetObjectMacro( Configuration, ConfigurationType );

  /** Set the database index of the instantiated elastix object. */
  virtual void SetDBIndex( DBIndexType _arg );
  virtual DBIndexType GetDBIndex( void )
  {
    return this->m_DBIndex;
  }

  /** Functions to get/set the ComponentDatabase
   * The component database contains pointers to functions
   * that create components.
   */
  elxGetObjectMacro( ComponentDatabase, ComponentDatabaseType );
  elxSetObjectMacro( ComponentDatabase, ComponentDatabaseType );

  /** Get the component containers.
   * The component containers store components, such as
   * the metric, in the form of an itk::Object::Pointer.
   */
  elxGetObjectMacro( RegistrationContainer, ObjectContainerType );
  elxGetObjectMacro( FixedImagePyramidContainer, ObjectContainerType );
  elxGetObjectMacro( MovingImagePyramidContainer, ObjectContainerType );
  elxGetObjectMacro( InterpolatorContainer, ObjectContainerType );
  elxGetObjectMacro( ImageSamplerContainer, ObjectContainerType );
  elxGetObjectMacro( MetricContainer, ObjectContainerType );
  elxGetObjectMacro( OptimizerContainer, ObjectContainerType );
  elxGetObjectMacro( ResamplerContainer, ObjectContainerType );
  elxGetObjectMacro( ResampleInterpolatorContainer, ObjectContainerType );
  elxGetObjectMacro( TransformContainer, ObjectContainerType );

  /** Set the component containers.
   * The component containers store components, such as
   * the metric, in the form of an itk::Object::Pointer.
   */
  elxSetObjectMacro( RegistrationContainer, ObjectContainerType );
  elxSetObjectMacro( FixedImagePyramidContainer, ObjectContainerType );
  elxSetObjectMacro( MovingImagePyramidContainer, ObjectContainerType );
  elxSetObjectMacro( InterpolatorContainer, ObjectContainerType );
  elxSetObjectMacro( ImageSamplerContainer, ObjectContainerType );
  elxSetObjectMacro( MetricContainer, ObjectContainerType );
  elxSetObjectMacro( OptimizerContainer, ObjectContainerType );
  elxSetObjectMacro( ResamplerContainer, ObjectContainerType );
  elxSetObjectMacro( ResampleInterpolatorContainer, ObjectContainerType );
  elxSetObjectMacro( TransformContainer, ObjectContainerType );

  /** Set/Get the fixed/moving image containers. */
  elxGetObjectMacro( FixedImageContainer, DataObjectContainerType );
  elxGetObjectMacro( MovingImageContainer, DataObjectContainerType );
  elxSetObjectMacro( FixedImageContainer, DataObjectContainerType );
  elxSetObjectMacro( MovingImageContainer, DataObjectContainerType );

  /** Set/Get the fixed/moving mask containers. */
  elxGetObjectMacro( FixedMaskContainer, DataObjectContainerType );
  elxGetObjectMacro( MovingMaskContainer, DataObjectContainerType );
  elxSetObjectMacro( FixedMaskContainer, DataObjectContainerType );
  elxSetObjectMacro( MovingMaskContainer, DataObjectContainerType );

  /** Set/Get the result image container. */
  elxGetObjectMacro( ResultImageContainer, DataObjectContainerType );
  elxSetObjectMacro( ResultImageContainer, DataObjectContainerType );

  /** Set/Get The Image FileName containers.
   * Normally, these are filled in the BeforeAllBase function.
   */
  elxGetObjectMacro( FixedImageFileNameContainer, FileNameContainerType );
  elxGetObjectMacro( MovingImageFileNameContainer, FileNameContainerType );
  elxSetObjectMacro( FixedImageFileNameContainer, FileNameContainerType );
  elxSetObjectMacro( MovingImageFileNameContainer, FileNameContainerType );

  /** Set/Get The Mask FileName containers.
   * Normally, these are filled in the BeforeAllBase function.
   */
  elxGetObjectMacro( FixedMaskFileNameContainer, FileNameContainerType );
  elxGetObjectMacro( MovingMaskFileNameContainer, FileNameContainerType );
  elxSetObjectMacro( FixedMaskFileNameContainer, FileNameContainerType );
  elxSetObjectMacro( MovingMaskFileNameContainer, FileNameContainerType );

  /** Define some convenience functions: GetNumberOfMetrics() for example. */
  elxGetNumberOfMacro( Registration );
  elxGetNumberOfMacro( FixedImagePyramid );
  elxGetNumberOfMacro( MovingImagePyramid );
  elxGetNumberOfMacro( Interpolator );
  elxGetNumberOfMacro( ImageSampler );
  elxGetNumberOfMacro( Metric );
  elxGetNumberOfMacro( Optimizer );
  elxGetNumberOfMacro( Resampler );
  elxGetNumberOfMacro( ResampleInterpolator );
  elxGetNumberOfMacro( Transform );
  elxGetNumberOfMacro( FixedImage );
  elxGetNumberOfMacro( MovingImage );
  elxGetNumberOfMacro( FixedImageFileName );
  elxGetNumberOfMacro( MovingImageFileName );
  elxGetNumberOfMacro( FixedMask );
  elxGetNumberOfMacro( MovingMask );
  elxGetNumberOfMacro( FixedMaskFileName );
  elxGetNumberOfMacro( MovingMaskFileName );
  elxGetNumberOfMacro( ResultImage );

  /** Set/Get the initial transform
   * The type is ObjectType, but the pointer should actually point
   * to an itk::Transform type (or inherited from that one).
   */
  elxSetObjectMacro( InitialTransform, ObjectType );
  elxGetObjectMacro( InitialTransform, ObjectType );

  /** Set/Get the final transform
   * The type is ObjectType, but the pointer should actually point
   * to an itk::Transform type (or inherited from that one).
   * You can use this to set it as an initial transform in another
   * ElastixBase instantiation.
   */
  elxSetObjectMacro( FinalTransform, ObjectType );
  elxGetObjectMacro( FinalTransform, ObjectType );

  /** Empty Run()-function to be overridden. */
  virtual int Run( void ) = 0;

  /** Empty ApplyTransform()-function to be overridden. */
  virtual int ApplyTransform( void ) = 0;

  /** Function that is called at the very beginning of ElastixTemplate::Run().
   * It checks the command line input arguments.
   */
  virtual int BeforeAllBase( void );

  /** Function that is called at the very beginning of ElastixTemplate::ApplyTransform().
   * It checks the command line input arguments.
   */
  virtual int BeforeAllTransformixBase( void );

  /** Functions called before and after registration.
   * They install/uninstall the xout["iteration"] field.
   */
  virtual void BeforeRegistrationBase( void );
  virtual void AfterRegistrationBase( void );

  /** Get the default precision of xout.
   * (The value assumed when no DefaultOutputPrecision is given in the
   * parameter file.
   */
  virtual int GetDefaultOutputPrecision( void ) const
  {
    return this->m_DefaultOutputPrecision;
  }

  /** Get whether direction cosines should be taken into account (true)
   * or ignored (false). This depends on the UseDirectionCosines
   * parameter. */
  virtual bool GetUseDirectionCosines( void ) const;

  /** Set/Get the original fixed image direction as a flat array
   * (d11 d21 d31 d21 d22 etc ) */
  virtual void SetOriginalFixedImageDirectionFlat(
    const FlatDirectionCosinesType & arg );
  virtual const FlatDirectionCosinesType &
    GetOriginalFixedImageDirectionFlat( void ) const;

  /** Creates transformation parameters map. */
  virtual void CreateTransformParametersMap( void ) = 0;

  /** Gets transformation parameters map. */
  virtual ParameterMapType GetTransformParametersMap( void ) const = 0;

  /** Set configuration vector. Library only. */
  virtual void SetConfigurations( std::vector< ConfigurationPointer > & configurations ) = 0;

protected:

  ElastixBase();
  virtual ~ElastixBase() {};

  ConfigurationPointer      m_Configuration;
  DBIndexType               m_DBIndex;
  ComponentDatabasePointer  m_ComponentDatabase;

  FlatDirectionCosinesType     m_OriginalFixedImageDirection;

  /** Convenient mini class to load the files specified by a filename container
   * The function GenerateImageContainer can be used without instantiating an
   * object of this class, since it is static. It has 2 arguments: the
   * fileNameContainer, and a string containing a short description of the images
   * to be loaded. In case of errors, an itk::ExceptionObject is thrown that
   * includes this short description and the fileName which caused the error.
   * See ElastixTemplate::Run() for an example of usage.
   *
   * The useDirection option is built in as a means to ignore the direction
   * cosines. Set it to false to force the direction cosines to identity.
   * The original direction cosines are returned separately.
   */
  template < class TImage >
  class MultipleImageLoader
  {
  public:
    typedef TImage                                       ImageType;
    typedef typename ImageType::Pointer                  ImagePointer;
    typedef itk::ImageFileReader<ImageType>              ImageReaderType;
    typedef typename ImageReaderType::Pointer            ImageReaderPointer;
    typedef typename ImageType::DirectionType            DirectionType;
    typedef itk::ChangeInformationImageFilter<ImageType> ChangeInfoFilterType;
    typedef typename ChangeInfoFilterType::Pointer       ChangeInfoFilterPointer;

    static DataObjectContainerPointer GenerateImageContainer(
      FileNameContainerType * fileNameContainer, const std::string & imageDescription,
      bool useDirectionCosines, DirectionType * originalDirectionCosines = NULL )
    {
      DataObjectContainerPointer imageContainer = DataObjectContainerType::New();

      /** Loop over all image filenames. */
      for ( unsigned int i = 0; i < fileNameContainer->Size(); ++i )
      {
        /** Setup reader. */
        ImageReaderPointer imageReader = ImageReaderType::New();
        imageReader->SetFileName( fileNameContainer->ElementAt( i ).c_str() );
        ChangeInfoFilterPointer infoChanger = ChangeInfoFilterType::New();
        DirectionType direction;
        direction.SetIdentity();
        infoChanger->SetOutputDirection( direction );
        infoChanger->SetChangeDirection( !useDirectionCosines );
        infoChanger->SetInput( imageReader->GetOutput() );

        /** Do the reading. */
        try
        {
          infoChanger->Update();
        }
        catch( itk::ExceptionObject & excp )
        {
          /** Add information to the exception. */
          std::string err_str = excp.GetDescription();
          err_str += "\nError occurred while reading the image described as "
            + imageDescription + ", with file name " + imageReader->GetFileName() + "\n";
          excp.SetDescription( err_str );
          /** Pass the exception to the caller of this function. */
          throw excp;
        }

        /** Store loaded image in the image container, as a DataObjectPointer. */
        ImagePointer image = infoChanger->GetOutput();
        imageContainer->CreateElementAt(i) = image.GetPointer();

        /** Store the original direction cosines */
        if( originalDirectionCosines )
        {
          *originalDirectionCosines = imageReader->GetOutput()->GetDirection();
        }

      } // end for i

      return imageContainer;

    } // end static method GenerateImageContainer

    /** Static method overloaded GenerateImageContainer. */
    static DataObjectContainerPointer GenerateImageContainer( DataObjectPointer image )
    {
      /** Allocate image container pointer. */
      DataObjectContainerPointer imageContainer = DataObjectContainerType::New();

      /** Store image in image container (for now only one image in container!). */
      imageContainer->CreateElementAt( 0 ) = image;

      /** Return the pointer to the new image container. */
      return imageContainer;

    } // GenerateImageContainer()

    MultipleImageLoader(){};
    ~MultipleImageLoader(){};

  }; // end class MultipleImageLoader

  class MultipleDataObjectFiller
  {
  public:

    /** GenerateImageContainer. */
    static DataObjectContainerPointer GenerateImageContainer(
      DataObjectPointer image )
    {
      unsigned int j = 0; //container with only one image for now

      /** Allocate image container pointer. */
      DataObjectContainerPointer imageContainer = DataObjectContainerType::New();

      /** Store image in image container. */
      imageContainer->CreateElementAt(j) = image;

      /** Return the pointer to the new image container. */
      return imageContainer;
    } // end GenerateImageContainer()

    /** Constructor and destructor. */
    MultipleDataObjectFiller(){};
    ~MultipleDataObjectFiller(){};
  }; // end class MultipleDataObjectFiller

private:

  ElastixBase( const Self& );     // purposely not implemented
  void operator=( const Self& );  // purposely not implemented

  xl::xoutrow_type      m_IterationInfo;

  int m_DefaultOutputPrecision;

  /** The component containers. These containers contain
   * SmartPointer's to itk::Object.
   */
  ObjectContainerPointer m_FixedImagePyramidContainer;
  ObjectContainerPointer m_MovingImagePyramidContainer;
  ObjectContainerPointer m_InterpolatorContainer;
  ObjectContainerPointer m_ImageSamplerContainer;
  ObjectContainerPointer m_MetricContainer;
  ObjectContainerPointer m_OptimizerContainer;
  ObjectContainerPointer m_RegistrationContainer;
  ObjectContainerPointer m_ResamplerContainer;
  ObjectContainerPointer m_ResampleInterpolatorContainer;
  ObjectContainerPointer m_TransformContainer;

  /** The Image and Mask containers. These are stored as pointers to itk::DataObject. */
  DataObjectContainerPointer m_FixedImageContainer;
  DataObjectContainerPointer m_MovingImageContainer;
  DataObjectContainerPointer m_FixedMaskContainer;
  DataObjectContainerPointer m_MovingMaskContainer;

  /** The result image container. These are stored as pointers to itk::DataObject. */
  DataObjectContainerPointer m_ResultImageContainer;

  /** The image and mask FileNameContainers. */
  FileNameContainerPointer    m_FixedImageFileNameContainer;
  FileNameContainerPointer    m_MovingImageFileNameContainer;
  FileNameContainerPointer    m_FixedMaskFileNameContainer;
  FileNameContainerPointer    m_MovingMaskFileNameContainer;

  /** The initial and final transform. */
  ObjectPointer m_InitialTransform;
  ObjectPointer m_FinalTransform;

  /** Use or ignore direction cosines. */
  bool m_UseDirectionCosines;

  /** Read a series of command line options that satisfy the following syntax:
   * {-f,-f0} \<filename0\> [-f1 \<filename1\> [ -f2 \<filename2\> ... ] ]
   *
   * This function is used by BeforeAllBase, and is not meant be used
   * at other locations. The errorcode remains the input value if no errors
   * occur. It is set to errorcode | 1 if the option was not given.
   */
  FileNameContainerPointer GenerateFileNameContainer(
    const std::string & optionkey,
    int & errorcode,
    bool printerrors,
    bool printinfo ) const;

};  // end class ElastixBase


} // end namespace elastix

#undef elxGetObjectMacro
#undef elxSetObjectMacro
#undef elxGetNumberOfMacro

#endif // end #ifndef __elxElastixBase_h


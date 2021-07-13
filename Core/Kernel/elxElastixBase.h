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

/**
 * This file contains the declaration of the elx::ElastixBase class.
 * elx::ElastixTemplate<> inherits from this class. It is an abstract class,
 * since it contains pure virtual functions (which must be implemented
 * in ElastixTemplate<>).
 *
 * The Configuration object is stored in this class.
 */

#ifndef elxElastixBase_h
#define elxElastixBase_h

#include "elxBaseComponent.h"
#include "elxComponentDatabase.h"
#include "elxConfiguration.h"
#include "elxMacro.h"
#include "xoutmain.h"

// ITK header files:
#include <itkChangeInformationImageFilter.h>
#include <itkDataObject.h>
#include <itkImageFileReader.h>
#include <itkObject.h>
#include <itkTimeProbe.h>
#include <itkVectorContainer.h>

#include <fstream>
#include <iomanip>

/** Like itkGet/SetObjectMacro, but in these macros the itkDebugMacro is
 * not called. Besides, they are not virtual, since
 * for now we do not need to override them somewhere.
 *
 * These macros are undef'd at the end of this file
 */
#define elxGetObjectMacro(_name, _type)                                                                                \
  _type * Get##_name(void) const { return this->m_##_name.GetPointer(); }
// end elxGetObjectMacro

#define elxSetObjectMacro(_name, _type)                                                                                \
  void Set##_name(_type * _arg)                                                                                        \
  {                                                                                                                    \
    if (this->m_##_name != _arg)                                                                                       \
    {                                                                                                                  \
      this->m_##_name = _arg;                                                                                          \
      this->itk::Object::Modified();                                                                                   \
    }                                                                                                                  \
  }
// end elxSetObjectMacro

/** defines for example: GetNumberOfMetrics() */
#define elxGetNumberOfMacro(_name)                                                                                     \
  unsigned int GetNumberOf##_name##s(void) const                                                                       \
  {                                                                                                                    \
    if (this->m_##_name##Container != nullptr)                                                                         \
    {                                                                                                                  \
      return this->m_##_name##Container->Size();                                                                       \
    }                                                                                                                  \
    return 0;                                                                                                          \
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

class ElastixBase
  : public itk::Object
  , public BaseComponent
{
public:
  /** Standard typedefs etc. */
  typedef ElastixBase   Self;
  typedef BaseComponent Superclass;

  /** Typedefs used in this class. */
  typedef Configuration                                         ConfigurationType;
  typedef ConfigurationType::Pointer                            ConfigurationPointer;
  typedef itk::Object                                           ObjectType; // for the components
  typedef ObjectType::Pointer                                   ObjectPointer;
  typedef itk::DataObject                                       DataObjectType; // for the images
  typedef DataObjectType::Pointer                               DataObjectPointer;
  typedef itk::VectorContainer<unsigned int, ObjectPointer>     ObjectContainerType;
  typedef ObjectContainerType::Pointer                          ObjectContainerPointer;
  typedef itk::VectorContainer<unsigned int, DataObjectPointer> DataObjectContainerType;
  typedef DataObjectContainerType::Pointer                      DataObjectContainerPointer;
  typedef itk::VectorContainer<unsigned int, std::string>       FileNameContainerType;
  typedef FileNameContainerType::Pointer                        FileNameContainerPointer;

  /** Result image */
  typedef itk::DataObject ResultImageType;

  /** Result deformation field */
  typedef itk::DataObject ResultDeformationFieldType;

  /** Other typedef's. */
  typedef ComponentDatabase                ComponentDatabaseType;
  typedef ComponentDatabaseType::Pointer   ComponentDatabasePointer;
  typedef ComponentDatabaseType::IndexType DBIndexType;
  typedef std::vector<double>              FlatDirectionCosinesType;

  /** Type for representation of the transform coordinates. */
  typedef double CoordRepType; // itk::CostFunction::ParametersValueType

  /** Typedef that is used in the elastix dll version. */
  typedef itk::ParameterMapInterface::ParameterMapType ParameterMapType;

  /** Typedef's for Timer class. */
  typedef itk::TimeProbe TimerType;

  /** Set/Get the Configuration Object. */
  elxGetObjectMacro(Configuration, ConfigurationType);
  elxSetObjectMacro(Configuration, ConfigurationType);

  /** Set the database index of the instantiated elastix object. */
  void
  SetDBIndex(DBIndexType _arg);

  DBIndexType
  GetDBIndex(void)
  {
    return this->m_DBIndex;
  }

  /** Get the component containers.
   * The component containers store components, such as
   * the metric, in the form of an itk::Object::Pointer.
   */
  elxGetObjectMacro(RegistrationContainer, ObjectContainerType);
  elxGetObjectMacro(FixedImagePyramidContainer, ObjectContainerType);
  elxGetObjectMacro(MovingImagePyramidContainer, ObjectContainerType);
  elxGetObjectMacro(InterpolatorContainer, ObjectContainerType);
  elxGetObjectMacro(ImageSamplerContainer, ObjectContainerType);
  elxGetObjectMacro(MetricContainer, ObjectContainerType);
  elxGetObjectMacro(OptimizerContainer, ObjectContainerType);
  elxGetObjectMacro(ResamplerContainer, ObjectContainerType);
  elxGetObjectMacro(ResampleInterpolatorContainer, ObjectContainerType);
  elxGetObjectMacro(TransformContainer, ObjectContainerType);

  /** Set the component containers.
   * The component containers store components, such as
   * the metric, in the form of an itk::Object::Pointer.
   */
  elxSetObjectMacro(RegistrationContainer, ObjectContainerType);
  elxSetObjectMacro(FixedImagePyramidContainer, ObjectContainerType);
  elxSetObjectMacro(MovingImagePyramidContainer, ObjectContainerType);
  elxSetObjectMacro(InterpolatorContainer, ObjectContainerType);
  elxSetObjectMacro(ImageSamplerContainer, ObjectContainerType);
  elxSetObjectMacro(MetricContainer, ObjectContainerType);
  elxSetObjectMacro(OptimizerContainer, ObjectContainerType);
  elxSetObjectMacro(ResamplerContainer, ObjectContainerType);
  elxSetObjectMacro(ResampleInterpolatorContainer, ObjectContainerType);
  elxSetObjectMacro(TransformContainer, ObjectContainerType);

  /** Set/Get the fixed/moving image containers. */
  elxGetObjectMacro(FixedImageContainer, DataObjectContainerType);
  elxGetObjectMacro(MovingImageContainer, DataObjectContainerType);
  elxSetObjectMacro(FixedImageContainer, DataObjectContainerType);
  elxSetObjectMacro(MovingImageContainer, DataObjectContainerType);

  /** Set/Get the fixed/moving mask containers. */
  elxGetObjectMacro(FixedMaskContainer, DataObjectContainerType);
  elxGetObjectMacro(MovingMaskContainer, DataObjectContainerType);
  elxSetObjectMacro(FixedMaskContainer, DataObjectContainerType);
  elxSetObjectMacro(MovingMaskContainer, DataObjectContainerType);

  /** Set/Get the result image container. */
  elxGetObjectMacro(ResultImageContainer, DataObjectContainerType);
  elxSetObjectMacro(ResultImageContainer, DataObjectContainerType);

  /** Set/Get the result image container. */
  elxGetObjectMacro(ResultDeformationFieldContainer, DataObjectContainerType);
  elxSetObjectMacro(ResultDeformationFieldContainer, DataObjectContainerType);

  /** Set/Get The Image FileName containers.
   * Normally, these are filled in the BeforeAllBase function.
   */
  elxGetObjectMacro(FixedImageFileNameContainer, FileNameContainerType);
  elxGetObjectMacro(MovingImageFileNameContainer, FileNameContainerType);
  elxSetObjectMacro(FixedImageFileNameContainer, FileNameContainerType);
  elxSetObjectMacro(MovingImageFileNameContainer, FileNameContainerType);

  /** Set/Get The Mask FileName containers.
   * Normally, these are filled in the BeforeAllBase function.
   */
  elxGetObjectMacro(FixedMaskFileNameContainer, FileNameContainerType);
  elxGetObjectMacro(MovingMaskFileNameContainer, FileNameContainerType);
  elxSetObjectMacro(FixedMaskFileNameContainer, FileNameContainerType);
  elxSetObjectMacro(MovingMaskFileNameContainer, FileNameContainerType);

  /** Define some convenience functions: GetNumberOfMetrics() for example. */
  elxGetNumberOfMacro(Registration);
  elxGetNumberOfMacro(FixedImagePyramid);
  elxGetNumberOfMacro(MovingImagePyramid);
  elxGetNumberOfMacro(Interpolator);
  elxGetNumberOfMacro(ImageSampler);
  elxGetNumberOfMacro(Metric);
  elxGetNumberOfMacro(Optimizer);
  elxGetNumberOfMacro(Resampler);
  elxGetNumberOfMacro(ResampleInterpolator);
  elxGetNumberOfMacro(Transform);
  elxGetNumberOfMacro(FixedImage);
  elxGetNumberOfMacro(MovingImage);
  elxGetNumberOfMacro(FixedImageFileName);
  elxGetNumberOfMacro(MovingImageFileName);
  elxGetNumberOfMacro(FixedMask);
  elxGetNumberOfMacro(MovingMask);
  elxGetNumberOfMacro(FixedMaskFileName);
  elxGetNumberOfMacro(MovingMaskFileName);
  elxGetNumberOfMacro(ResultImage);
  elxGetNumberOfMacro(ResultDeformationField);

  /** Set/Get the initial transform
   * The type is ObjectType, but the pointer should actually point
   * to an itk::Transform type (or inherited from that one).
   */
  elxSetObjectMacro(InitialTransform, ObjectType);
  elxGetObjectMacro(InitialTransform, ObjectType);

  /** Set/Get the final transform
   * The type is ObjectType, but the pointer should actually point
   * to an itk::Transform type (or inherited from that one).
   * You can use this to set it as an initial transform in another
   * ElastixBase instantiation.
   */
  elxSetObjectMacro(FinalTransform, ObjectType);
  elxGetObjectMacro(FinalTransform, ObjectType);

  /** Empty Run()-function to be overridden. */
  virtual int
  Run(void) = 0;

  /** Empty ApplyTransform()-function to be overridden. */
  virtual int
  ApplyTransform(void) = 0;

  /** Function that is called at the very beginning of ElastixTemplate::Run().
   * It checks the command line input arguments.
   */
  int
  BeforeAllBase(void) override;

  /** Function that is called at the very beginning of ElastixTemplate::ApplyTransform().
   * It checks the command line input arguments.
   */
  int
  BeforeAllTransformixBase(void);

  /** Function called before registration.
   * It installs the IterationInfo field.
   */
  void
  BeforeRegistrationBase(void) override;

  ResultImageType *
  GetResultImage(const unsigned int idx = 0) const;

  void
  SetResultImage(DataObjectPointer result_image);

  ResultDeformationFieldType *
  GetResultDeformationField(unsigned int idx = 0) const;

  void
  SetResultDeformationField(DataObjectPointer result_deformationfield);


  /** Get the default precision of xout.
   * (The value assumed when no DefaultOutputPrecision is given in the
   * parameter file.
   */
  int
  GetDefaultOutputPrecision(void) const
  {
    return this->m_DefaultOutputPrecision;
  }


  /** Get whether direction cosines should be taken into account (true)
   * or ignored (false). This depends on the UseDirectionCosines
   * parameter. */
  bool
  GetUseDirectionCosines(void) const;

  /** Set/Get the original fixed image direction as a flat array
   * (d11 d21 d31 d21 d22 etc ) */
  void
  SetOriginalFixedImageDirectionFlat(const FlatDirectionCosinesType & arg);

  const FlatDirectionCosinesType &
  GetOriginalFixedImageDirectionFlat(void) const;

  /** Creates transformation parameters map. */
  virtual void
  CreateTransformParametersMap(void) = 0;

  /** Gets transformation parameters map. */
  ParameterMapType
  GetTransformParametersMap(void) const;

  /** Set configuration vector. Library only. */
  void
  SetConfigurations(const std::vector<ConfigurationPointer> & configurations);

  /** Return configuration from vector of configurations. Library only. */
  ConfigurationPointer
  GetConfiguration(const size_t index) const;

  xl::xoutrow &
  GetIterationInfo(void)
  {
    return m_IterationInfo;
  }

  xl::xoutbase &
  GetIterationInfoAt(const char * const name)
  {
    return m_IterationInfo[name];
  }

  void
  AddTargetCellToIterationInfo(const char * const name)
  {
    m_IterationInfo.AddTargetCell(name);
  }

protected:
  ElastixBase();
  ~ElastixBase() override = default;

  ConfigurationPointer m_Configuration;
  DBIndexType          m_DBIndex;

  FlatDirectionCosinesType m_OriginalFixedImageDirection;

  /** Timers. */
  TimerType m_Timer0{};
  TimerType m_IterationTimer{};
  TimerType m_ResolutionTimer{};

  /** Store the CurrentTransformParameterFileName. */
  std::string m_CurrentTransformParameterFileName;

  /** A vector of configuration objects, needed when transformix is used as library. */
  std::vector<ConfigurationPointer> m_Configurations;

  /** Count the number of iterations. */
  unsigned int m_IterationCounter{};

  /** Stores transformation parameters map. */
  ParameterMapType m_TransformParametersMap;

  std::ofstream m_IterationInfoFile;

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
  template <class TImage>
  class ITK_TEMPLATE_EXPORT MultipleImageLoader
  {
  public:
    typedef typename TImage::DirectionType DirectionType;

    static DataObjectContainerPointer
    GenerateImageContainer(const FileNameContainerType * const fileNameContainer,
                           const std::string &                 imageDescription,
                           bool                                useDirectionCosines,
                           DirectionType *                     originalDirectionCosines = nullptr)
    {
      const auto imageContainer = DataObjectContainerType::New();

      /** Loop over all image filenames. */
      for (const auto & fileName : *fileNameContainer)
      {
        /** Setup reader. */
        const auto imageReader = itk::ImageFileReader<TImage>::New();
        imageReader->SetFileName(fileName);
        const auto    infoChanger = itk::ChangeInformationImageFilter<TImage>::New();
        DirectionType direction;
        direction.SetIdentity();
        infoChanger->SetOutputDirection(direction);
        infoChanger->SetChangeDirection(!useDirectionCosines);
        infoChanger->SetInput(imageReader->GetOutput());

        /** Do the reading. */
        try
        {
          infoChanger->Update();
        }
        catch (itk::ExceptionObject & excp)
        {
          /** Add information to the exception. */
          std::string err_str = excp.GetDescription();
          err_str += "\nError occurred while reading the image described as " + imageDescription + ", with file name " +
                     imageReader->GetFileName() + "\n";
          excp.SetDescription(err_str);
          /** Pass the exception to the caller of this function. */
          throw excp;
        }

        /** Store loaded image in the image container, as a DataObjectPointer. */
        const auto image = infoChanger->GetOutput();
        imageContainer->push_back(image);

        /** Store the original direction cosines */
        if (originalDirectionCosines != nullptr)
        {
          *originalDirectionCosines = imageReader->GetOutput()->GetDirection();
        }

      } // end for

      return imageContainer;

    } // end static method GenerateImageContainer


    MultipleImageLoader() = default;
    ~MultipleImageLoader() = default;
  };

  /** Generates a container that contains the specified data object */
  static DataObjectContainerPointer
  GenerateDataObjectContainer(DataObjectPointer dataObject);

private:
  ElastixBase(const Self &) = delete;
  void
  operator=(const Self &) = delete;

  xl::xoutrow m_IterationInfo;

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

  /** The result deformation field container. These are stored as pointers to itk::DataObject. */
  DataObjectContainerPointer m_ResultDeformationFieldContainer;

  /** The image and mask FileNameContainers. */
  FileNameContainerPointer m_FixedImageFileNameContainer;
  FileNameContainerPointer m_MovingImageFileNameContainer;
  FileNameContainerPointer m_FixedMaskFileNameContainer;
  FileNameContainerPointer m_MovingMaskFileNameContainer;

  /** The initial and final transform. */
  ObjectPointer m_InitialTransform;
  ObjectPointer m_FinalTransform;

  /** Use or ignore direction cosines. */
  bool m_UseDirectionCosines;
};

} // end namespace elastix

#undef elxGetObjectMacro
#undef elxSetObjectMacro
#undef elxGetNumberOfMacro

#endif // end #ifndef elxElastixBase_h

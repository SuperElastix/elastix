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
#include "elxDefaultConstruct.h"
#include "elxIterationInfo.h"
#include "elxMacro.h"
#include "elxlog.h"

// ITK header files:
#include <itkChangeInformationImageFilter.h>
#include <itkDataObject.h>
#include <itkImageFileReader.h>
#include <itkMersenneTwisterRandomVariateGenerator.h>
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
#define elxGetObjectMacro(_name, _type) \
  _type * Get##_name() const { return m_##_name.GetPointer(); }
// end elxGetObjectMacro

#define elxSetObjectMacro(_name, _type) \
  void Set##_name(_type * _arg)         \
  {                                     \
    if (m_##_name != _arg)              \
    {                                   \
      m_##_name = _arg;                 \
      this->itk::Object::Modified();    \
    }                                   \
  }
// end elxSetObjectMacro

/** defines for example: GetNumberOfMetrics() */
#define elxGetNumberOfMacro(_name)           \
  unsigned int GetNumberOf##_name##s() const \
  {                                          \
    if (m_##_name##Container != nullptr)     \
    {                                        \
      return m_##_name##Container->Size();   \
    }                                        \
    return 0;                                \
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
  ITK_DISALLOW_COPY_AND_MOVE(ElastixBase);

  /** Standard typedefs etc. */
  using Self = ElastixBase;
  using Superclass = BaseComponent;

  /** Typedefs used in this class. */
  using ObjectPointer = itk::Object::Pointer;
  using DataObjectPointer = itk::DataObject::Pointer; // for the images
  using ObjectContainerType = itk::VectorContainer<unsigned int, ObjectPointer>;
  using ObjectContainerPointer = ObjectContainerType::Pointer;
  using DataObjectContainerType = itk::VectorContainer<unsigned int, DataObjectPointer>;
  using DataObjectContainerPointer = DataObjectContainerType::Pointer;
  using FileNameContainerType = itk::VectorContainer<unsigned int, std::string>;
  using FileNameContainerPointer = FileNameContainerType::Pointer;

  /** Result image */
  using ResultImageType = itk::DataObject;

  /** Result deformation field */
  using ResultDeformationFieldType = itk::DataObject;

  /** Other typedef's. */
  using ComponentDatabasePointer = ComponentDatabase::Pointer;
  using DBIndexType = ComponentDatabase::IndexType;
  using FlatDirectionCosinesType = std::vector<double>;

  /** Type for representation of the transform coordinates. */
  using CoordinateType = double; // itk::CostFunction::ParametersValueType

  /** Typedef that is used in the elastix dll version. */
  using ParameterMapType = itk::ParameterMapInterface::ParameterMapType;

  /** Set/Get the Configuration Object. */
  elxGetObjectMacro(Configuration, Configuration);
  elxSetObjectMacro(Configuration, Configuration);

  /** Set the database index of the instantiated elastix object. */
  void
  SetDBIndex(DBIndexType _arg);

  DBIndexType
  GetDBIndex()
  {
    return m_DBIndex;
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

  /** Set/Get the fixed/moving points (landmarks, used by the CorrespondingPointsEuclideanDistance metric). */
  elxGetObjectMacro(FixedPoints, const itk::Object);
  elxGetObjectMacro(MovingPoints, const itk::Object);
  elxSetObjectMacro(FixedPoints, const itk::Object);
  elxSetObjectMacro(MovingPoints, const itk::Object);

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
   * The type is itk::Object, but the pointer should actually point
   * to an itk::Transform type (or inherited from that one).
   */
  elxSetObjectMacro(InitialTransform, itk::Object);
  elxGetObjectMacro(InitialTransform, itk::Object);

  /** Set/Get the final transform
   * The type is itk::Object, but the pointer should actually point
   * to an itk::Transform type (or inherited from that one).
   * You can use this to set it as an initial transform in another
   * ElastixBase instantiation.
   */
  elxSetObjectMacro(FinalTransform, itk::Object);
  elxGetObjectMacro(FinalTransform, itk::Object);

  /** Empty Run()-function to be overridden. */
  virtual int
  Run() = 0;

  /** Empty ApplyTransform()-function to be overridden. */
  virtual int
  ApplyTransform(bool doReadTransform) = 0;

  /** Function that is called at the very beginning of ElastixTemplate::Run().
   * It checks the command line input arguments.
   */
  int
  BeforeAllBase() override;

  /** Function that is called at the very beginning of ElastixTemplate::ApplyTransform().
   * It checks the command line input arguments.
   */
  int
  BeforeAllTransformixBase();

  itk::Statistics::MersenneTwisterRandomVariateGenerator &
  GetRandomVariateGenerator()
  {
    return m_RandomVariateGenerator;
  }
  ResultImageType *
  GetResultImage(const unsigned int idx = 0) const;

  void
  SetResultImage(DataObjectPointer result_image);

  ResultDeformationFieldType *
  GetResultDeformationField(unsigned int idx = 0) const;

  void
  SetResultDeformationField(DataObjectPointer result_deformationfield);


  /** Get whether direction cosines should be taken into account (true)
   * or ignored (false). This depends on the UseDirectionCosines
   * parameter. */
  bool
  GetUseDirectionCosines() const;

  /** Set/Get the original fixed image direction as a flat array
   * (d11 d21 d31 d21 d22 etc ) */
  void
  SetOriginalFixedImageDirectionFlat(const FlatDirectionCosinesType & arg);

  const FlatDirectionCosinesType &
  GetOriginalFixedImageDirectionFlat() const;

  /** Creates transformation parameters map. */
  virtual void
  CreateTransformParameterMap() = 0;

  /** Gets transformation parameters map. */
  ParameterMapType
  GetTransformParameterMap() const;

  /** Set configuration vector. Library only. */
  void
  SetTransformConfigurations(const std::vector<Configuration::ConstPointer> & configurations);

  /** Return configuration from vector of transformation configurations. Library only. */
  Configuration::ConstPointer
  GetTransformConfiguration(const size_t index) const;

  /** Returns the transformation configuration just before the specified one, or null. Returns null if the specified
   * configuration does not have a previous one in the vector of transformation configurations. */
  Configuration::ConstPointer
  GetPreviousTransformConfiguration(const Configuration & configuration) const;

  /** Returns the number of configurations in the vector of transformation configurations. */
  size_t
  GetNumberOfTransformConfigurations() const;

  IterationInfo &
  GetIterationInfo()
  {
    return m_IterationInfo;
  }

  std::ostream &
  GetIterationInfoAt(const char * const name)
  {
    return m_IterationInfo[name];
  }

  void
  AddTargetCellToIterationInfo(const char * const name)
  {
    m_IterationInfo.AddNewTargetCell(name);
  }

protected:
  ElastixBase();
  ~ElastixBase() override = default;

  DBIndexType m_DBIndex{ 0 };

  FlatDirectionCosinesType m_OriginalFixedImageDirectionFlat;

  /** Timers. */
  itk::TimeProbe m_Timer0{};
  itk::TimeProbe m_IterationTimer{};
  itk::TimeProbe m_ResolutionTimer{};

  /** Store the CurrentTransformParameterFileName. */
  std::string m_CurrentTransformParameterFileName;

  /** Count the number of iterations. */
  unsigned int m_IterationCounter{};

  /** Stores transformation parameters map. */
  ParameterMapType m_TransformParameterMap;

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
  template <typename TImage>
  class ITK_TEMPLATE_EXPORT MultipleImageLoader
  {
  public:
    using DirectionType = typename TImage::DirectionType;

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
        const auto infoChanger = itk::ChangeInformationImageFilter<TImage>::New();
        infoChanger->SetChangeDirection(!useDirectionCosines);

        /** Do the reading. */
        try
        {
          const auto image = itk::ReadImage<TImage>(fileName);
          infoChanger->SetInput(image);
          infoChanger->Update();

          /** Store the original direction cosines */
          if (originalDirectionCosines != nullptr)
          {
            *originalDirectionCosines = image->GetDirection();
          }
        }
        catch (itk::ExceptionObject & excp)
        {
          /** Add information to the exception. */
          std::string err_str = excp.GetDescription();
          err_str += "\nError occurred while reading the image described as " + imageDescription + ", with file name " +
                     fileName + "\n";
          excp.SetDescription(err_str);
          /** Pass the exception to the caller of this function. */
          throw;
        }

        /** Store loaded image in the image container, as a DataObjectPointer. */
        imageContainer->push_back(infoChanger->GetOutput());


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
  Configuration::Pointer m_Configuration{ nullptr };

  /** A vector of configuration objects, needed when transformix is used as library. */
  std::vector<Configuration::ConstPointer> m_TransformConfigurations;

  IterationInfo m_IterationInfo;

  /** The component containers. These containers contain
   * SmartPointer's to itk::Object.
   */
  ObjectContainerPointer m_FixedImagePyramidContainer{ ObjectContainerType::New() };
  ObjectContainerPointer m_MovingImagePyramidContainer{ ObjectContainerType::New() };
  ObjectContainerPointer m_InterpolatorContainer{ ObjectContainerType::New() };
  ObjectContainerPointer m_ImageSamplerContainer{ ObjectContainerType::New() };
  ObjectContainerPointer m_MetricContainer{ ObjectContainerType::New() };
  ObjectContainerPointer m_OptimizerContainer{ ObjectContainerType::New() };
  ObjectContainerPointer m_RegistrationContainer{ ObjectContainerType::New() };
  ObjectContainerPointer m_ResamplerContainer{ ObjectContainerType::New() };
  ObjectContainerPointer m_ResampleInterpolatorContainer{ ObjectContainerType::New() };
  ObjectContainerPointer m_TransformContainer{ ObjectContainerType::New() };

  /** The Image and Mask containers. These are stored as pointers to itk::DataObject. */
  DataObjectContainerPointer m_FixedImageContainer{ DataObjectContainerType::New() };
  DataObjectContainerPointer m_MovingImageContainer{ DataObjectContainerType::New() };
  DataObjectContainerPointer m_FixedMaskContainer{ DataObjectContainerType::New() };
  DataObjectContainerPointer m_MovingMaskContainer{ DataObjectContainerType::New() };

  /** The fixed and moving points (landmarks, used by the CorrespondingPointsEuclideanDistance metric). */
  itk::SmartPointer<const itk::Object> m_FixedPoints{ nullptr };
  itk::SmartPointer<const itk::Object> m_MovingPoints{ nullptr };

  /** The result image container. These are stored as pointers to itk::DataObject. */
  DataObjectContainerPointer m_ResultImageContainer{ DataObjectContainerType::New() };

  /** The result deformation field container. These are stored as pointers to itk::DataObject. */
  DataObjectContainerPointer m_ResultDeformationFieldContainer;

  /** The image and mask FileNameContainers. */
  FileNameContainerPointer m_FixedImageFileNameContainer{ FileNameContainerType::New() };
  FileNameContainerPointer m_MovingImageFileNameContainer{ FileNameContainerType::New() };
  FileNameContainerPointer m_FixedMaskFileNameContainer{ FileNameContainerType::New() };
  FileNameContainerPointer m_MovingMaskFileNameContainer{ FileNameContainerType::New() };

  /** The initial and final transform. */
  ObjectPointer m_InitialTransform{ nullptr };
  ObjectPointer m_FinalTransform{ nullptr };

  /** Use or ignore direction cosines.
   * From Elastix 4.3 to 4.7: Ignore direction cosines by default, for
   * backward compatability. From Elastix 4.8: set it to true by default. */
  bool m_UseDirectionCosines{ true };

  elx::DefaultConstruct<itk::Statistics::MersenneTwisterRandomVariateGenerator> m_RandomVariateGenerator{};
};

} // end namespace elastix

#undef elxGetObjectMacro
#undef elxSetObjectMacro
#undef elxGetNumberOfMacro

#endif // end #ifndef elxElastixBase_h

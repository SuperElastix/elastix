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
#ifndef __elxOpenCLOpenCLFixedGenericPyramid_h
#define __elxOpenCLOpenCLFixedGenericPyramid_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "elxFixedGenericPyramid.h"
#include "itkGPUImage.h"

namespace elastix
{

/**
 * \class OpenCLFixedGenericPyramid
 * \brief A pyramid based on the itk::GenericMultiResolutionPyramidImageFilter.
 * The parameters used in this class are:
 * \parameter Pyramid: Select this pyramid as follows:\n
 *    <tt>(FixedImagePyramid "OpenCLFixedGenericImagePyramid")</tt>
 * \parameter Pyramid: Enable the OpenCL pyramid as follows:\n
 *    <tt>(OpenCLFixedGenericImagePyramidUseOpenCL "true")</tt>
 *
 * \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
 * Department of Radiology, Leiden, The Netherlands
 *
 * \note This work was funded by the Netherlands Organisation for
 * Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
 *
 * \sa FixedGenericPyramid
 * \ingroup ImagePyramids
 */

template< class TElastix >
class OpenCLFixedGenericPyramid :
  public FixedGenericPyramid< TElastix >
{
public:

  /** Standard ITK-stuff. */
  typedef OpenCLFixedGenericPyramid                             Self;
  typedef FixedGenericPyramid< TElastix >                       Superclass;
  typedef typename FixedGenericPyramid< TElastix >::Superclass1 Superclass1;
  typedef typename FixedGenericPyramid< TElastix >::Superclass2 Superclass2;
  typedef itk::SmartPointer< Self >                             Pointer;
  typedef itk::SmartPointer< const Self >                       ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( OpenCLFixedGenericPyramid, FixedGenericPyramid );

  /** Name of this class.
   * Use this name in the parameter file to select this specific pyramid. \n
   * example: <tt>(FixedImagePyramid "OpenCLFixedGenericImagePyramid")</tt>\n
   */
  elxClassNameMacro( "OpenCLFixedGenericImagePyramid" );

  /** Get the ImageDimension. */
  itkStaticConstMacro( ImageDimension, unsigned int, Superclass1::ImageDimension );

  /** Typedefs inherited from the superclass. */
  typedef typename Superclass1::InputImageType             InputImageType;
  typedef typename Superclass1::OutputImageType            OutputImageType;
  typedef typename Superclass1::InputImageType::PixelType  InputImagePixelType;
  typedef typename Superclass1::OutputImageType::PixelType OutputImagePixelType;

  /** Typedefs for factory. */
  typedef typename itk::ObjectFactoryBase::Pointer ObjectFactoryBasePointer;

  /** GPU Typedefs for GPU image and GPU filter. */
  typedef itk::GPUImage< InputImagePixelType, InputImageType::ImageDimension >
    GPUInputImageType;
  typedef typename GPUInputImageType::Pointer GPUInputImagePointer;
  typedef itk::GPUImage< OutputImagePixelType, OutputImageType::ImageDimension >
    GPUOutputImageType;

  typedef itk::GenericMultiResolutionPyramidImageFilter<
    GPUInputImageType,
    GPUOutputImageType, float > GPUPyramidType;
  typedef typename GPUPyramidType::Pointer GPUPyramidPointer;

  /** Do some things before registration. */
  virtual void BeforeRegistration( void );

  /** Function to read parameters from a file. */
  virtual void ReadFromFile( void );

protected:

  /** This method performs all configuration for GPU pyramid. */
  void BeforeGenerateData( void );

  /** Executes GPU pyramid. */
  virtual void GenerateData( void );

  /** The constructor. */
  OpenCLFixedGenericPyramid();
  /** The destructor. */
  virtual ~OpenCLFixedGenericPyramid() {}

private:

  /** The private constructor. */
  OpenCLFixedGenericPyramid( const Self & ); // purposely not implemented
  /** The private copy constructor. */
  void operator=( const Self & ); // purposely not implemented

  /** Register/Unregister factories. */
  void RegisterFactories( void );

  void UnregisterFactories( void );

  /** Helper method to report switching to CPU mode. */
  void SwitchingToCPUAndReport( const bool configError );

  /** Helper method to report to elastix log. */
  void ReportToLog( void );

  GPUPyramidPointer                       m_GPUPyramid;
  bool                                    m_GPUPyramidReady;
  bool                                    m_GPUPyramidCreated;
  bool                                    m_ContextCreated;
  bool                                    m_UseOpenCL;
  std::vector< ObjectFactoryBasePointer > m_Factories;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxOpenCLFixedGenericPyramid.hxx"
#endif

#endif // end #ifndef __elxOpenCLOpenCLFixedGenericPyramid_h

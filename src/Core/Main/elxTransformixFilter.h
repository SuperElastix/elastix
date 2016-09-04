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
#ifndef elxTransformixFilter_h
#define elxTransformixFilter_h

#include "itkImageSource.h"

#include "elxTransformixMain.h"
#include "elxParameterObject.h"
#include "elxPixelType.h"

/**
 * \class TransformixFilter
 * \brief ITK Filter interface to the Transformix library.
 */

namespace elastix
{

template< typename TMovingImage >
class ELASTIXLIB_API TransformixFilter : public itk::ImageSource< TMovingImage >
{
public:

  /** Standard ITK typedefs. */
  typedef TransformixFilter               Self;
  typedef itk::ImageSource< TMovingImage > Superclass;
  typedef itk::SmartPointer< Self >       Pointer;
  typedef itk::SmartPointer< const Self > ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( Self, itk::ImageSource );

  /** Typedefs. */
  typedef elastix::TransformixMain             TransformixMainType;
  typedef TransformixMainType::Pointer         TransformixMainPointer;
  typedef TransformixMainType::ArgumentMapType ArgumentMapType;
  typedef ArgumentMapType::value_type          ArgumentMapEntryType;

  typedef itk::ProcessObject::DataObjectIdentifierType    DataObjectIdentifierType;
  typedef TransformixMainType::DataObjectContainerType    DataObjectContainerType;
  typedef TransformixMainType::DataObjectContainerPointer DataObjectContainerPointer;

  typedef ParameterObject                               ParameterObjectType;
  typedef ParameterObjectType::ParameterMapVectorType   ParameterMapVectorType;
  typedef ParameterObjectType::ParameterMapType         ParameterMapType;
  typedef ParameterObjectType::ParameterValueVectorType ParameterValueVectorType;
  typedef typename ParameterObjectType::Pointer         ParameterObjectPointer;
  typedef typename ParameterObjectType::ConstPointer    ParameterObjectConstPointer;

  typedef typename TMovingImage::Pointer      InputImagePointer;
  typedef typename TMovingImage::ConstPointer InputImageConstPointer;

  itkStaticConstMacro( MovingImageDimension, unsigned int, TMovingImage::ImageDimension );
  itkGetStaticConstMacro( MovingImageDimension );

  /** Set/Get/Add moving image. */
  virtual void SetMovingImage( TMovingImage * inputImage );
  InputImageConstPointer GetMovingImage( void );
  virtual void RemoveMovingImage( void );

  /** Set/Get/Remove moving point set filename. */
  itkSetMacro( FixedPointSetFileName, std::string );
  itkGetMacro( FixedPointSetFileName, std::string );
  virtual void RemoveFixedPointSetFileName() { this->SetFixedPointSetFileName( "" ); }

  /** Compute spatial Jacobian On/Off. */
  itkSetMacro( ComputeSpatialJacobian, bool );
  itkGetConstMacro( ComputeSpatialJacobian, bool );
  itkBooleanMacro( ComputeSpatialJacobian );

  /** Compute determinant of spatial Jacobian On/Off. */
  itkSetMacro( ComputeDeterminantOfSpatialJacobian, bool );
  itkGetConstMacro( ComputeDeterminantOfSpatialJacobian, bool );
  itkBooleanMacro( ComputeDeterminantOfSpatialJacobian );

  /** Compute deformation field On/Off. */
  itkSetMacro( ComputeDeformationField, bool );
  itkGetConstMacro( ComputeDeformationField, bool );
  itkBooleanMacro( ComputeDeformationField );

  /** Get/Set transform parameter object. */
  virtual void SetTransformParameterObject( ParameterObjectPointer transformParameterObject );

  ParameterObjectType * GetTransformParameterObject( void );

  const ParameterObjectType * GetTransformParameterObject( void ) const;

  /** Set/Get/Remove output directory. */
  itkSetMacro( OutputDirectory, std::string );
  itkGetConstMacro( OutputDirectory, std::string );
  virtual void RemoveOutputDirectory() { this->SetOutputDirectory( "" ); }

  /** Set/Get/Remove log filename. */
  virtual void SetLogFileName( std::string logFileName );

  itkGetConstMacro( LogFileName, std::string );
  virtual void RemoveLogFileName( void );

  /** Log to std::cout on/off. */
  itkSetMacro( LogToConsole, bool );
  itkGetConstMacro( LogToConsole, bool );
  itkBooleanMacro( LogToConsole );

  /** Log to file on/off. */
  itkSetMacro( LogToFile, bool );
  itkGetConstMacro( LogToFile, bool );
  itkBooleanMacro( LogToFile );

protected:

  TransformixFilter( void );

  virtual void GenerateData( void ) ITK_OVERRIDE;

private:

  TransformixFilter( const Self & ); // purposely not implemented
  void operator=( const Self & );    // purposely not implemented

  /** IsEmpty. */
  virtual bool IsEmpty( const InputImagePointer inputImage );

  /** Let transformix handle input verification internally */
  virtual void VerifyInputInformation( void ) ITK_OVERRIDE {};

  /** Tell the compiler we want all definitions of Get/Set/Remove
   *  from ProcessObject and TransformixFilter. */
  using itk::ProcessObject::SetInput;
  using itk::ProcessObject::GetInput;
  using itk::ProcessObject::RemoveInput;

  std::string m_FixedPointSetFileName;
  bool        m_ComputeSpatialJacobian;
  bool        m_ComputeDeterminantOfSpatialJacobian;
  bool        m_ComputeDeformationField;

  std::string m_OutputDirectory;
  std::string m_LogFileName;

  bool m_LogToConsole;
  bool m_LogToFile;

};

} // namespace elx

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxTransformixFilter.hxx"
#endif

#endif // elxTransformixFilter_h

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
 * \brief Transformix library exposed as an ITK filter.
 */

namespace elastix {

template< typename TInputImage >
class TransformixFilter : public itk::ImageSource< TInputImage >
{
public:

  /** Standard ITK typedefs. */
  typedef TransformixFilter               Self;
  typedef itk::ImageSource< TInputImage > Superclass; 
  typedef itk::SmartPointer< Self >       Pointer;
  typedef itk::SmartPointer< const Self > ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( Self, itk::ImageSource );

  /** Typedefs. */
  typedef elastix::TransformixMain                          TransformixMainType;
  typedef TransformixMainType::Pointer                      TransformixMainPointer;
  typedef TransformixMainType::ArgumentMapType              ArgumentMapType;
  typedef ArgumentMapType::value_type                       ArgumentMapEntryType;

  typedef itk::ProcessObject::DataObjectIdentifierType      DataObjectIdentifierType;
  typedef TransformixMainType::DataObjectContainerType      DataObjectContainerType;
  typedef TransformixMainType::DataObjectContainerPointer   DataObjectContainerPointer;

  typedef ParameterObject::ParameterMapVectorType           ParameterMapVectorType;
  typedef ParameterObject::ParameterMapType                 ParameterMapType;
  typedef ParameterObject::ParameterValueVectorType         ParameterValueVectorType;
  typedef typename ParameterObject::Pointer                 ParameterObjectPointer;
  typedef typename ParameterObject::ConstPointer            ParameterObjectConstPointer;

  typedef typename TInputImage::Pointer                     InputImagePointer;
  typedef typename TInputImage::ConstPointer                InputImageConstPointer;

  itkStaticConstMacro( InputImageDimension, unsigned int, TInputImage::ImageDimension );

  /** Set/Get/Add moving image. */
  void SetInput( TInputImage* inputImage );
  InputImageConstPointer GetInput( void );
  void RemoveInput( void );

  /** Set/Get/Remove moving point set filename. */
  itkSetMacro( InputPointSetFileName, std::string );
  itkGetMacro( InputPointSetFileName, std::string );
  void RemoveInputPointSetFileName() { this->SetInputPointSetFileName( "" ); };

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
  void SetTransformParameterObject( ParameterObjectPointer parameterObject );
  ParameterObjectPointer GetTransformParameterObject( void );

  /** Set/Get/Remove output directory. */
  itkSetMacro( OutputDirectory, std::string );
  itkGetConstMacro( OutputDirectory, std::string );
  void RemoveOutputDirectory() { this->SetOutputDirectory( "" ); };

  /** Set/Get/Remove log filename. */
  void SetLogFileName( std::string logFileName );
  itkGetConstMacro( LogFileName, std::string );
  void RemoveLogFileName( void );

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

  void GenerateData( void ) ITK_OVERRIDE;

private:

  TransformixFilter( const Self & ); // purposely not implemented
  void operator=( const Self & );    // purposely not implemented

  /** IsEmpty. */
  bool IsEmpty( const InputImagePointer inputImage );

  /** Let transformix handle input verification internally */
  void VerifyInputInformation( void ) ITK_OVERRIDE {};

  /** Tell the compiler we want all definitions of Get/Set/Remove 
   *  from ProcessObject and TransformixFilter. */
  using itk::ProcessObject::SetInput;
  using itk::ProcessObject::GetInput;
  using itk::ProcessObject::RemoveInput;

  std::string   m_InputPointSetFileName;
  bool          m_ComputeSpatialJacobian;
  bool          m_ComputeDeterminantOfSpatialJacobian;
  bool          m_ComputeDeformationField;

  std::string   m_OutputDirectory;
  std::string   m_LogFileName;

  bool          m_LogToConsole;
  bool          m_LogToFile;

};

} // namespace elx

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxTransformixFilter.hxx"
#endif

#endif // elxTransformixFilter_h

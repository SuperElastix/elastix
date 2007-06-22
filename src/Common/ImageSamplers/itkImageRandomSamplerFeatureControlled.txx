#ifndef __ImageRandomSamplerFeatureControlled_txx
#define __ImageRandomSamplerFeatureControlled_txx

#include "itkImageRandomSamplerFeatureControlled.h"

#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkContinuousIndex.h"
#include "itkNumericTraits.h"

namespace itk
{

  /**
  * ******************* Constructor *******************
  */

  template< class TInputImage >
    ImageRandomSamplerFeatureControlled< TInputImage >
    ::ImageRandomSamplerFeatureControlled()
  {
    this->m_NumberOfSamples = 100;
    /** Setup random generator. */
    this->m_RandomGenerator = RandomGeneratorType::New();
    //this->m_RandomGenerator->Initialize();
    this->m_InternalFullSampler = InternalFullSamplerType::New();

    this->m_TreeNeedsUpdate = true;
    this->m_ListSampleNeedsUpdate = true;
    this->m_NumberOfFeatureImages = 0;
    this->m_Tree = TreeType::New();
    this->m_TreeSearch = TreeSearchType::New();
    this->m_ListSample = ListSampleType::New();

    this->m_SplittingRule = "ANN_KD_SL_MIDPT";
    this->m_ShrinkingRule = "ANN_BD_SIMPLE";
    this->m_BucketSize = 5;
    this->m_ErrorBound = 1.0;
    this->m_UseXYZAsFeatures = false;

  } // end Constructor


  /**
  * ********************* SetListSampleNeedsUpdate ****************************
  */

  template< class TInputImage >
    void ImageRandomSamplerFeatureControlled< TInputImage >
    ::SetListSampleNeedsUpdate( bool _arg )
  {
    if ( this->m_ListSampleNeedsUpdate != _arg )
    {
      this->m_ListSampleNeedsUpdate = _arg;
      this->Modified();
      if ( _arg )
      {
        this->SetTreeNeedsUpdate( true );
      }
    }
  } // end SetListSampleNeedsUpdate


  /**
  * ********************* SetNumberOfFeatureImages ****************************
  */

  template< class TInputImage >
    void ImageRandomSamplerFeatureControlled< TInputImage >
    ::SetNumberOfFeatureImages( unsigned int arg )
  {
    if ( this->m_NumberOfFeatureImages != arg )
    {
      this->m_FeatureImages.resize( arg );
      this->m_NumberOfFeatureImages = arg;
      this->Modified();
      this->SetListSampleNeedsUpdate( true );
    }

  } // end SetNumberOfFeatureImage


  /**
  * ********************* GetTotalNumberOfFeatures ****************************
  */

  template< class TInputImage >
    unsigned int ImageRandomSamplerFeatureControlled< TInputImage >
    ::GetTotalNumberOfFeatures(void) const
  {
    if ( this->GetUseXYZAsFeatures() )
    {
      return ( this->GetNumberOfFeatureImages() + InputImageDimension );
    }
    else
    {
      return this->GetNumberOfFeatureImages();
    }
  } // end GetTotalNumberOfFeatures


  /**
  * ********************* SetFeatureImage ****************************
  */

  template< class TInputImage >
    void ImageRandomSamplerFeatureControlled< TInputImage >
    ::SetFeatureImage( unsigned int i, FeatureImageType * im )
  {
    if ( i + 1 > this->m_NumberOfFeatureImages )
    {
      this->m_FeatureImages.resize( i + 1 );
      this->m_FeatureImages[ i ] = im;
      this->m_NumberOfFeatureImages = i;
      this->Modified();
      this->SetListSampleNeedsUpdate( true );
    }
    else
    {
      if ( this->m_FeatureImages[ i ] != im )
      {
        this->m_FeatureImages[ i ] = im;
        this->Modified();
        this->SetListSampleNeedsUpdate( true );
      }
    }

  } // end SetFeatureImage


  /**
  * ********************* GetFeatureImage ****************************
  */

  template< class TInputImage >
    const typename ImageRandomSamplerFeatureControlled< TInputImage >
    ::FeatureImageType *
    ImageRandomSamplerFeatureControlled< TInputImage >
    ::GetFeatureImage( unsigned int i ) const
  {
    return this->m_FeatureImages[ i ].GetPointer();
  } // end GetFeatureImage


  /**
  * ******************* SetBucketSize *******************
  */

  template< class TInputImage >
    void ImageRandomSamplerFeatureControlled< TInputImage >
    ::SetBucketSize(unsigned int _arg)
  {
    itkDebugMacro( "setting BucketSize to " << _arg );
    if ( this->m_BucketSize != _arg )
    {
      this->m_BucketSize = _arg;
      this->Modified();
      this->SetTreeNeedsUpdate( true );
    }
  } // end SetBucketSize


  /**
  * ******************* SetSplittingRule *******************
  */

  template< class TInputImage >
    void ImageRandomSamplerFeatureControlled< TInputImage >
    ::SetSplittingRule(const char * _arg)
  {
    if ( _arg && (_arg == this->m_SplittingRule) )
    {
      return;
    }
    if (_arg)
    {
      this->m_SplittingRule = _arg;
    }
    else
    {
      this->m_SplittingRule = "";
    }
    this->Modified();
    this->SetTreeNeedsUpdate( true );
  } // end SetSplittingRule


  /**
  * ******************* SetShrinkingRule *******************
  */

  template< class TInputImage >
    void ImageRandomSamplerFeatureControlled< TInputImage >
    ::SetShrinkingRule(const char * _arg)
  {
    if ( _arg && (_arg == this->m_ShrinkingRule) )
    {
      return;
    }
    if (_arg)
    {
      this->m_ShrinkingRule = _arg;
    }
    else
    {
      this->m_ShrinkingRule = "";
    }
    this->Modified();
    this->SetTreeNeedsUpdate( true );
  } // end SetShrinkingRule


  /**
  * ******************* SetUseXYZAsFeatures *******************
  */

  template< class TInputImage >
    void ImageRandomSamplerFeatureControlled< TInputImage >
    ::SetUseXYZAsFeatures( bool _arg )
  {
    itkDebugMacro( "setting UseXYZAsFeatures to " << _arg );
    if ( this->m_UseXYZAsFeatures != _arg )
    {
      this->m_UseXYZAsFeatures = _arg;
      this->Modified();
      this->SetListSampleNeedsUpdate( true );
    }
  } // end SetUseXYZAsFeatures


  /**
  * ******************* UpdateTree *******************
  */

  template< class TInputImage >
    void
    ImageRandomSamplerFeatureControlled< TInputImage >
    ::UpdateTree(void)
  {
    /** Set some parameters: */
    this->m_Tree->SetSample( this->m_ListSample );

    this->m_Tree->SetBucketSize( this->m_BucketSize );
    this->m_Tree->SetSplittingRule( this->m_SplittingRule );

    /** bdTree does not work with only one feature */
    if ( this->GetTotalNumberOfFeatures() == 1 )
    {
      std::string noshrink = "ANN_BD_NONE";
      this->m_Tree->SetShrinkingRule( noshrink );
    }
    else
    {
      this->m_Tree->SetShrinkingRule( this->m_ShrinkingRule );
    }

    /** Generate the tree */
    this->m_Tree->GenerateTree();

    /** The tree is up-to-date now. */
    this->m_TreeNeedsUpdate = false;

  } // end UpdateTree  


  /**
  * ******************* UpdateListSample *****************
  */

  template< class TInputImage >
    void
    ImageRandomSamplerFeatureControlled< TInputImage >
    ::UpdateListSample(void)
  {
    /** Index typedefs */
    typedef InputImageIndexType::IndexValueType    IndexValueType;
    typedef ContinuousIndex<double, InputImageDimension>  CIndexType;

    /** Iterator typedefs */    
    typedef ImageRegionConstIteratorWithIndex<FeatureImageType> FeatureImageIteratorType;
    typedef std::vector< FeatureImageIteratorType >             FeatureImageIteratorArray;
    typedef typename ImageSampleContainerType::ConstIterator    SampleContainerIteratorType;

    /** Some convenience variables, which will be used a lot in this function */
    typename ImageSampleContainerType::Pointer allValidSamples =
      this->m_InternalFullSampler->GetOutput();
    InputImageConstPointer inputImage = this->GetInput();
    const unsigned long numberOfValidSamples = allValidSamples->Size();
    const unsigned int numberOfFeatureImages = this->GetNumberOfFeatureImages();
    const unsigned int numberOfFeatures = this->GetTotalNumberOfFeatures();
    
    /** Some check */
    if ( numberOfFeatures == 0 )
    {
      itkExceptionMacro( << "You need to set at least one feature image for this sampler." );
    }

    /** Set the size of the m_ListSample */    
    this->m_ListSample->SetMeasurementVectorSize( numberOfFeatures );
    this->m_ListSample->Resize( numberOfValidSamples );

    /** Setup the feature image iterators (fits) */
    FeatureImageIteratorArray fits;
    for ( unsigned int i = 0; i < numberOfFeatureImages; ++i )
    {
      if ( this->m_FeatureImages[i].IsNull() )
      {
        itkExceptionMacro( << "Feature image " << i << " has not been set properly!" );
      }
      else
      {
        fits.push_back( FeatureImageIteratorType(
          this->m_FeatureImages[i], this->GetInputImageRegion() ) );
        fits[i].GoToBegin();
      }
    }

    /** Set up the sample container iterator (sit) */
    SampleContainerIteratorType sit = allValidSamples->Begin();
    SampleContainerIteratorType sitend = allValidSamples->End();

    /** Initialize feature vector mean and standard deviation arrays 
     * These will be used to normalize the data to mean=0.0, standard dev=1.0 */
    FeatureVectorType meanFeatVec( numberOfFeatures );
    FeatureVectorType stdFeatVec( numberOfFeatures );
    meanFeatVec.Fill(0.0);
    stdFeatVec.Fill(0.0);

    /** loop over fits, until for every voxel in sit the features have been put
    * in the m_ListSample */
    unsigned int sitindex = 0;
    bool fitsAtEnd = false;
    if ( numberOfFeatureImages != 0 )
    {
      fitsAtEnd = fits[0].IsAtEnd();
    }
    while ( (sit != sitend) && !fitsAtEnd )
    {
      /** Get the cindex of the current sample in the samplecontainer. */
      const InputImagePointType & inputPoint = sit.Value().m_ImageCoordinates;
      CIndexType cindex;
      inputImage->TransformPhysicalPointToContinuousIndex( inputPoint, cindex );

      bool sameindex = true;
      if ( numberOfFeatureImages != 0 )
      {
        /** Get the index of the current voxel of featureimages[0] */
        const InputImageIndexType & index = fits[0].GetIndex();

        /** Check if they are the same */

        for ( unsigned int d = 0; d < InputImageDimension; ++d )
        {
          if (  index[d] != static_cast<IndexValueType>( vnl_math_rnd( cindex[d] ) )  )
          {
            sameindex = false;
            break; // this will usually happen at d==0 already!
          }
        }
      }

      /** If the same, add the Feature to the ListSample */
      if ( sameindex )
      {
        /** Compose feature vector */
        FeatureVectorType featvec( numberOfFeatures );
        for ( unsigned int f = 0; f < numberOfFeatureImages; ++f )
        {
          featvec[ f ] = fits[ f ].Get();    
        }
        /** Add xyz as features, if desired */
        unsigned int d = 0;
        for ( unsigned int f  = numberOfFeatureImages; f < numberOfFeatures; ++f )
        {
          
          featvec[ f ] = inputPoint[ d ];
          ++d;
        }

        /** mean += x_i,   std += (x_i).^2; later we will need those to compute mean and std */
        meanFeatVec += featvec;
        stdFeatVec += element_product( featvec, featvec );
                
        /** Store feature vector in m_ListSample */
        this->m_ListSample->SetMeasurementVector( sitindex, featvec );

        /** Next sample */
        ++sit;
        ++sitindex;
      }

      /** Next feature pixel */
      for ( unsigned int f = 0; f < numberOfFeatureImages; ++f )
      {
        ++( fits[f] );
      }    
      if ( numberOfFeatureImages != 0 )
      {
        fitsAtEnd = fits[0].IsAtEnd();
      }

    } // end loop over sample container

    /** This is not supposed to happen */
    if ( sitindex != numberOfValidSamples )
    {
      itkExceptionMacro( << "The ListSample's actual size is not equal to the size of the SampleContainer" );
    }

    /** Set the number of samples actually set. */
    this->m_ListSample->SetActualSize( numberOfValidSamples );
 
    /** Compute mean and std */
    const FeatureVectorValueType smallNumber =
      NumericTraits< FeatureVectorValueType >::epsilon() * 1000.0;
    meanFeatVec /= numberOfValidSamples;
    for ( unsigned int f = 0; f < numberOfFeatures; ++f )
    {
      stdFeatVec[f] -= numberOfValidSamples * meanFeatVec[f] * meanFeatVec[f];
      stdFeatVec[f] /= ( numberOfValidSamples - 1 );
      stdFeatVec[f] = vcl_sqrt( stdFeatVec[f] );
      
      if ( stdFeatVec[f] < smallNumber )
      {
        itkExceptionMacro( << "Feature " << f << " has (almost) zero standard deviation!" );
      }
    }
    
    /** Normalize each feature to mean=0.0 and std=1.0 */
    for ( unsigned int i = 0; i < numberOfValidSamples; ++i )
    {
      FeatureVectorType feat;
      this->m_ListSample->GetMeasurementVector( i, feat );
      for ( unsigned int f = 0; f < numberOfFeatures; ++f )
      {
        feat[f] = ( feat[f] - meanFeatVec[f] ) / stdFeatVec[f];
      }     
      this->m_ListSample->SetMeasurementVector( i, feat );
    }

    /** The m_ListSample is up-to-date now. */
    this->m_ListSampleNeedsUpdate = false;

  } // end UpdateTree  


  /**
  * ******************* GenerateData *******************
  */

  template< class TInputImage >
    void
    ImageRandomSamplerFeatureControlled< TInputImage >
    ::GenerateData( void )
  {
    typedef typename OutputVectorContainerType::ElementIdentifier SampleContainerIndexType;

    /** Get handles to the input image and output sample container. */
    InputImageConstPointer inputImage = this->GetInput();
    typename ImageSampleContainerType::Pointer sampleContainer = this->GetOutput();
    typename MaskType::Pointer mask = const_cast<MaskType *>( this->GetMask() );

    /** Make sure the internal full sampler is up-to-date. */
    const unsigned long mtime0 = this->m_InternalFullSampler->GetMTime();
    this->m_InternalFullSampler->SetInput(inputImage);
    this->m_InternalFullSampler->SetMask(mask);
    this->m_InternalFullSampler->SetInputImageRegion( this->GetInputImageRegion() );
    this->m_InternalFullSampler->Update();
    typename ImageSampleContainerType::Pointer allValidSamples =
      this->m_InternalFullSampler->GetOutput();
    unsigned long numberOfValidSamples = allValidSamples->Size();

    /** Check if the internal full sampler has been modified during update
    * If yes, the m_ListSample and m_Tree need to be update. */
    const unsigned long mtime1 = this->m_InternalFullSampler->GetMTime();
    if ( mtime1 > mtime0 )
    {
      this->m_TreeNeedsUpdate = true;
      this->m_ListSampleNeedsUpdate = true;
    }

    /** Update the ListSample, if necessary */
    if ( this->GetListSampleNeedsUpdate() )
    {
      this->UpdateListSample();
    }

    /** Update the tree, if necessary */
    if ( this->GetTreeNeedsUpdate() )
    {
      this->UpdateTree();
    }

    /** Configure the TreeSearch */
    this->m_TreeSearch->SetKNearestNeighbors( this->GetNumberOfSamples() );
    this->m_TreeSearch->SetErrorBound( this->GetErrorBound() );
    this->m_TreeSearch->SetBinaryTree( this->m_Tree );

    /** Take a random sample from the allValidSamples-container. */
    unsigned long randomIndex = 
      this->m_RandomGenerator->GetIntegerVariate( numberOfValidSamples );
    FeatureVectorType querypoint;
    this->m_ListSample->GetMeasurementVector( randomIndex, querypoint );
    
    /** Find its k nearest neighbours */
    IndexArrayType indices;
    DistanceArrayType distances;
    this->m_TreeSearch->Search( querypoint, indices, distances );
    

    /** Copy the selected samples from the allValidSamples container to 
    * the output sampleContainer */
    for ( unsigned int i = 0; i < indices.GetSize(); ++i )
    {
      sampleContainer->push_back(  allValidSamples->ElementAt( 
        static_cast<SampleContainerIndexType>( indices[i] ) )  );
    }


  } // end GenerateData


  /**
  * ******************* PrintSelf *******************
  */

  template< class TInputImage >
    void
    ImageRandomSamplerFeatureControlled< TInputImage >
    ::PrintSelf( std::ostream& os, Indent indent ) const
  {
    Superclass::PrintSelf( os, indent );

    os << indent << "NumberOfSamples: " << this->m_NumberOfSamples << std::endl;
    os << indent << "InternalFullSampler: " << this->m_InternalFullSampler.GetPointer() << std::endl;
    os << indent << "RandomGenerator: " << this->m_RandomGenerator.GetPointer() << std::endl;

    os << indent << "NumberOfFeatureImages: "  << this->m_NumberOfFeatureImages << std::endl;

    /** Print the feature image pointers. */
    for ( unsigned int i = 0; i < this->m_NumberOfFeatureImages; i++ )
    {
      os << indent << "FeatureImages[" << i << "]: "
        << this->m_FeatureImages[ i ].GetPointer() << std::endl;
    }

  } // end PrintSelf



} // end namespace itk

#endif // end #ifndef __ImageRandomSamplerFeatureControlled_txx


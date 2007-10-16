#ifndef __ImageRandomSamplerFeatureControlled_h
#define __ImageRandomSamplerFeatureControlled_h

#include "itkImageRandomSamplerBase.h"
#include "itkMersenneTwisterRandomVariateGenerator.h"
#include "itkImageFullSampler.h"

/** Includes for the kNN trees. */
#include "itkArray.h"
#include "itkListSampleCArray.h"
//#include "itkBinaryTreeBase.h"
//#include "itkBinaryTreeSearchBase.h"

/** Supported trees. */
#include "itkANNkDTree.h"
#include "itkANNbdTree.h"
//#include "itkANNBruteForceTree.h"

/** Supported tree searchers. */
//#include "itkANNStandardTreeSearch.h"
//#include "itkANNFixedRadiusTreeSearch.h"
#include "itkANNPriorityTreeSearch.h"

namespace itk
{

  /** \class ImageRandomSamplerFeatureControlled
   *
   * \brief Samples randomly some voxels of an image, guided by some feature images
   * 
   * This sampler needs additional feature images. 
   * A set of samples is selected, with features clustered around a randomly
   * chosen point in the feature space. This is realised using a kNN search.
   * 
   * This sampler is rather experimental.
   *
   * 
   */

  template < class TInputImage >
  class ImageRandomSamplerFeatureControlled :
    public ImageRandomSamplerBase< TInputImage >
  {
  public:

		/** Standard ITK-stuff. */
    typedef ImageRandomSamplerFeatureControlled     Self;
    typedef ImageRandomSamplerBase< TInputImage >   Superclass;
    typedef SmartPointer<Self>                Pointer;
    typedef SmartPointer<const Self>          ConstPointer;

		/** Method for creation through the object factory. */
    itkNewMacro( Self );

		/** Run-time type information (and related methods). */
    itkTypeMacro( ImageRandomSamplerFeatureControlled, ImageRandomSamplerBase );

		/** Typedefs inherited from the superclass. */
    typedef typename Superclass::DataObjectPointer            DataObjectPointer;
    typedef typename Superclass::OutputVectorContainerType    OutputVectorContainerType;
    typedef typename Superclass::OutputVectorContainerPointer OutputVectorContainerPointer;
    typedef typename Superclass::InputImageType               InputImageType;
    typedef typename Superclass::InputImagePointer            InputImagePointer;
    typedef typename Superclass::InputImageConstPointer       InputImageConstPointer;
    typedef typename Superclass::InputImageRegionType         InputImageRegionType;
    typedef typename Superclass::InputImagePixelType          InputImagePixelType;
    typedef typename Superclass::ImageSampleType              ImageSampleType;
    typedef typename Superclass::ImageSampleContainerType     ImageSampleContainerType;
    typedef typename Superclass::MaskType                     MaskType;

    /** The input image dimension. */
    itkStaticConstMacro( InputImageDimension, unsigned int,
			Superclass::InputImageDimension );

    /** Other typedefs. */
    typedef typename InputImageType::IndexType    InputImageIndexType;
    typedef typename InputImageType::PointType    InputImagePointType;
    typedef InputImageType                        FeatureImageType;
    typedef InputImagePointer                     FeatureImagePointer;
    typedef InputImageConstPointer                FeatureImageConstPointer;
    typedef InputImagePixelType                   FeatureImagePixelType;
    typedef std::vector< FeatureImagePointer >    FeatureImageContainerType;
           
    /** The random number generator used to generate random indices. */
    typedef itk::Statistics::MersenneTwisterRandomVariateGenerator RandomGeneratorType;

    /** Set the number of feature images. */
    void SetNumberOfFeatureImages( unsigned int arg );

    /** Get the number of feature images. */
    itkGetConstMacro( NumberOfFeatureImages, unsigned int );

    /** Get the total number of features. 
     * If m_UseXYZAsFeatures==false, this equals the NumberOfFeatureImages
     * if m_UseXYZAsFeatures==true, this equals (NumberOfFeatureImages+InputImageDimension)
     */
    virtual unsigned int GetTotalNumberOfFeatures(void) const;

    /** Set the feature images. */
    void SetFeatureImage( unsigned int i, FeatureImageType * im );
    void SetFeatureImage( FeatureImageType * im )
    {
      this->SetFeatureImage( 0, im );
    }

    /** Get the feature images. */
    const FeatureImageType * GetFeatureImage( unsigned int i ) const;
    const FeatureImageType * GetFeatureImage( void ) const
    {
      return this->GetFeatureImage( 0 );
    }
    
    
    /** Set/Get the error bound, a floating point number >= 0.0.
     * Default: 1.0 */
    itkGetConstMacro(ErrorBound, double);
    itkSetClampMacro(ErrorBound, double, 0.0, NumericTraits<double>::max() );  

    /** Set/Get the bucket size\n
     * Default: 5 \n
     * The tree will be recomputed in the next call to update after changing this parameter */
    itkGetConstMacro(BucketSize, unsigned int);
    virtual void SetBucketSize(unsigned int arg);    

    /** Set/Get the splitting rule\n
     * Choose one of \{ ANN_KD_STD, ANN_KD_MIDPT, ANN_KD_SL_MIDPT, ANN_KD_FAIR, ANN_KD_SL_FAIR, ANN_KD_SUGGEST \} \n
     * Default: "ANN_KD_SL_MIDPT"\n
     * The tree will be recomputed in the next call to update after changing this parameter */
    itkGetStringMacro(SplittingRule);
    virtual void SetSplittingRule(const char * _arg);
    virtual void SetSplittingRule(const std::string & _arg)
    {
      this->SetSplittingRule( _arg.c_str() );
    }

    /** Set/Get the shrinking rule\n
     * Choose one of \{ ANN_BD_NONE, ANN_BD_SIMPLE, ANN_BD_CENTROID, ANN_BD_SUGGEST \} \n
     * Default: "ANN_BD_CENTROID"\n
     * The tree will be recomputed in the next call to update after changing this parameter */
    itkGetStringMacro(ShrinkingRule);
    virtual void SetShrinkingRule(const char * _arg);
    virtual void SetShrinkingRule(const std::string & _arg)
    {
      this->SetShrinkingRule( _arg.c_str() );
    }  

    /** Set/Get whether the coordinates of each pixel should be used as features
     * default: false */
    itkGetConstMacro( UseXYZAsFeatures, bool );
    virtual void SetUseXYZAsFeatures( bool _arg );

    /** Set/Get whether the tree needs an update. 
     * This variable is set to true by the functions SetBucketSize, SetSplittingRule, 
     * SetShrinkingRule, SetUseXYZAsFeatures, SetFeatureImage, and SetNumberOfFeatureImages */
    itkGetConstMacro(TreeNeedsUpdate, bool);
    itkSetMacro(TreeNeedsUpdate, bool);

    /** Set/Get whether the ListSample (feature images in one huge array) needs an update. 
     * This variable is set to true by the functions SetFeatureImage, SetNumberOfFeatureImages,
     * and SetUseXYZAsFeatures. When set to true, also the m_TreeNeedsUpdate variable is set to true. */
    itkGetConstMacro(ListSampleNeedsUpdate, bool);
    virtual void SetListSampleNeedsUpdate( bool _arg );
       
  protected:

    /** Typedefs for KNN stuff. The ANNlib is only compiled for doubles, 
     * so we need to convert the FeatureImagePixelType to double */
    typedef double                                      FeatureVectorValueType;
    typedef Array< FeatureVectorValueType >             FeatureVectorType;
    typedef typename Statistics::ListSampleCArray<
      FeatureVectorType >                               ListSampleType;
    typedef ANNbdTree< ListSampleType >                 TreeType;
    typedef ANNPriorityTreeSearch< ListSampleType >     TreeSearchType;
    typedef typename TreeSearchType::IndexArrayType     IndexArrayType;
    typedef typename TreeSearchType::DistanceArrayType  DistanceArrayType;

    typedef itk::ImageFullSampler<InputImageType>       InternalFullSamplerType;
    
    /** Member variables */
    typename RandomGeneratorType::Pointer     m_RandomGenerator;
    typename InternalFullSamplerType::Pointer m_InternalFullSampler;
    bool                                      m_TreeNeedsUpdate;
    bool                                      m_ListSampleNeedsUpdate;
    unsigned int                              m_NumberOfFeatureImages;
    FeatureImageContainerType                 m_FeatureImages;
    typename TreeType::Pointer                m_Tree;
    typename TreeSearchType::Pointer          m_TreeSearch;
    typename ListSampleType::Pointer          m_ListSample;

    /** The constructor. */
    ImageRandomSamplerFeatureControlled();
    /** The destructor. */
    virtual ~ImageRandomSamplerFeatureControlled() {};

    /** PrintSelf. */
    void PrintSelf( std::ostream& os, Indent indent ) const;

    /** Function that does the work. */
    virtual void GenerateData( void );

    /** Update the ListSample. Assumes the InternalFullSampler has been updated already */
    virtual void UpdateListSample(void);

    /** Update the Tree. Assumes the ListSample has been updated already. */
    virtual void UpdateTree(void);
  
            
  private:

		/** The private constructor. */
    ImageRandomSamplerFeatureControlled( const Self& );	        // purposely not implemented
		/** The private copy constructor. */
    void operator=( const Self& );				    // purposely not implemented

    /** Member variables */
     std::string m_SplittingRule;
    std::string m_ShrinkingRule;
    unsigned int m_BucketSize;
    double m_ErrorBound;
    bool m_UseXYZAsFeatures;

  }; // end class ImageRandomSamplerFeatureControlled


} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkImageRandomSamplerFeatureControlled.txx"
#endif

#endif // end #ifndef __ImageRandomSamplerFeatureControlled_h

